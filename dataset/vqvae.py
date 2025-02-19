# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

from collections import namedtuple
from .LlamaModel import LlamaModel
from transformers.utils import (

    ModelOutput,
)
from transformers.models.llama.modeling_llama import (

    LlamaPreTrainedModel,
    
)
from transformers.modeling_utils import PreTrainedModel
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
base_model_dic={
    "llama":LlamaModel
}

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            output_size = inputs_size[:-1] + (inputs_size[-1]//embedding_size,)
            inputs_flatten = inputs.reshape(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*output_size)
            ctx.mark_non_differentiable(indices)
            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply

class VQEmbeddingMovingAverage(nn.Module):
    def __init__(self, K, D, slice=1, decay=0.99):
        super().__init__()
        embedding = torch.zeros(K, D//slice)
        embedding.uniform_(-1./K, 1./K)
        self.decay = decay
        self.slice = slice

        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.ones(K))
        self.register_buffer("ema_w", self.embedding.clone())

    def straight_through(self, z_e_x, data_parallel=False):
        K, D = self.embedding.size()

        z_e_x_ = z_e_x.contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding)
        z_q_x = z_q_x_.contiguous()

        if self.training and not data_parallel:
            #ema_w_slice = self.ema_w[:, slice_start:slice_end].clone()
            encodings = F.one_hot(indices, K).float()
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            dw = encodings.transpose(1, 0)@z_e_x_.reshape([-1, D])
            self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw

            self.embedding = self.ema_w / (self.ema_count.unsqueeze(-1))
            self.embedding = self.embedding.detach()
            self.ema_count = self.ema_count.detach()
            self.ema_w = self.ema_w.detach()

        z_q_x_bar_flatten = torch.index_select(self.embedding, dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.contiguous()

        return z_q_x, z_q_x_bar
    
    def ema_update(self, z_e_x):
        K, D = self.embedding.size()

        z_e_x_ = z_e_x.contiguous()
        _, indices = vq_st(z_e_x_, self.embedding)

        encodings = F.one_hot(indices, K).float()
        self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

        dw = encodings.transpose(1, 0)@z_e_x_.reshape([-1, D])
        self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw

        self.embedding = self.ema_w / (self.ema_count.unsqueeze(-1))
        self.embedding = self.embedding.detach()
        self.ema_count = self.ema_count.detach()
        self.ema_w = self.ema_w.detach()


class VQEmbedding(nn.Module):
    def __init__(self, K, D, slices=1):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)
        self.slice = slices

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.contiguous()
        latents = []
        for i in range(self.slice):
            latents.append(vq(z_e_x_, self.embedding.weight[:, i*self.embedding.weight.data.size(1)//self.slice:(i+1)*self.embedding.weight.data.size(1)//self.slice]))
        return torch.cat(latents, dim=-1)

    def straight_through(self, z_e_x, data_parallel=False):
        z_q_x_list = []
        z_q_x_bar_list = []
        for i in range(self.slice):
            slice_start = i*self.embedding.weight.data.size(1)//self.slice
            slice_end = (i+1)*self.embedding.weight.data.size(1)//self.slice
            z_e_x_ = z_e_x.contiguous()[..., slice_start:slice_end]
            z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight[:, slice_start:slice_end])
            z_q_x = z_q_x_.contiguous()

            z_q_x_bar_flatten = torch.index_select(self.embedding.weight, dim=0, index=indices)
            z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
            z_q_x_bar = z_q_x_bar_.contiguous()
            z_q_x_list.append(z_q_x)
            z_q_x_bar_list.append(z_q_x_bar)
        return torch.cat(z_q_x_list, dim=-1), torch.cat(z_q_x_bar_list, dim=-1)

class VQAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # if "data_parallel" in config:
        #     self.data_parallel = config.data_parallel
        # else:
        self.data_parallel = False 
        


        self.latent_step = config.latent_step
      
     

        self.encode_down_project = nn.Linear(config.base_hidden_size, config.hidden_size)
        self.encoder = base_model_dic[config.base_model_type](config)
        self.encoder_up_project = nn.Linear(config.hidden_size, config.base_hidden_size)

        self.decoder_down_project = nn.Linear(config.base_hidden_size, config.hidden_size)
        self.decoder = base_model_dic[config.base_model_type](config)
        
        
        # if "ma_update" in config and not (config.ma_update):
        self.codebook = VQEmbedding(config.K, config.base_hidden_size)
        self.ma_update = False
        # else:
        #     self.codebook = VQEmbeddingMovingAverage(config.K, config.trajectory_embd, self.code_per_step)
        #     self.ma_update = True

        self.latent_pooling = nn.MaxPool1d(self.latent_step, stride=self.latent_step)
       

        self.predict = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
      


    def encode(self, joined_inputs,attention_mask):
        
        b, t, joined_dimension = joined_inputs.size()
        


        # forward the GPT model
        x = self.encode_down_project(joined_inputs)
        x = self.encoder(inputs_embeds=x,attention_mask=attention_mask).last_hidden_state
        x = self.encoder_up_project(x)
        ## [ B x T x embedding_dim ]


       
        x = self.latent_pooling(x.transpose(1, 2)).transpose(1, 2)
      
        return x

    def decode(self, latents,attention_mask):
        """
            latents: [B x (T//self.latent_step*self.code_per_step) x latent_size]
            state: [B x observation_dimension]
        """
        B, T, _ = latents.shape
        latents = self.decoder_down_project(latents)
       
        latents = torch.repeat_interleave(latents, self.latent_step, dim=1)
        

        
        

        x = self.decoder(inputs_embeds=latents,attention_mask=attention_mask).last_hidden_state

        ## [B x T x obs_dim]
        joined_pred = self.predict(x)
       
        return joined_pred

    def forward(self, joined_inputs,attention_mask):
        trajectory_feature = self.encode(joined_inputs,attention_mask)
        latents_st, latents = self.codebook.straight_through(trajectory_feature, data_parallel=self.data_parallel)
        
        joined_pred = self.decode(latents_st,attention_mask)
        return joined_pred, latents, trajectory_feature

@dataclass
class CausalVQVAEOutputWithPast(ModelOutput):
    
   

    reconstructed: Optional[torch.FloatTensor] = None
    feature_masked: Optional[torch.FloatTensor] = None
    reconstruction_loss: Optional[torch.FloatTensor] = None
    loss_vq: Optional[torch.FloatTensor] = None
    loss_commit: Optional[torch.FloatTensor] = None
    


class VQVAE(LlamaPreTrainedModel):
    """  the full Llama language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # if "data_parallel" in config:
        #     self.data_parallel = config.data_parallel
        # else:
        self.data_parallel = False 
        # input embedding stem (+1 for stop token)
        self.model = VQAutoencoder(config)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.latent_pooling = nn.MaxPool1d(config.latent_step, stride=config.latent_step)
        
        self.K = config.K
        self.vocab_size = config.vocab_size
        self.latent_step = config.latent_step
        
    
        self.apply(self._init_weights)
      

    def get_last_layer(self):
        return self.model.predict.weight

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    @torch.no_grad()
    def encode(self, input_ids, attention_mask,vocab_size):
        joined_inputs=self.embed_tokens(input_ids)
        b, t, joined_dimension = joined_inputs.size()
        

        trajectory_feature = self.model.encode(torch.cat([joined_inputs, masks], dim=2))
        if self.model.ma_update:
            indices = vq(trajectory_feature, self.model.codebook.embedding)
        else:
            indices = vq(trajectory_feature, self.model.codebook.embedding.weight)
    
        latent_mask = self.latent_pooling(attention_mask.unsqueeze(1).to(joined_inputs.dtype)).unsqueeze(1)
        indices = indices.reshape(b, -1)
        indices += vocab_size
        indices=indices.masked_fill(latent_mask==0,-100).tolist()
        indices = [x[:x.index(-100)] for x in indices]
        return indices

    def decode(self, latent, attention_mask):
        """
        Decode a trajectory from feature vectors.
        Args:
            latent: [B x T x D] latent feature vectors
            state: [B x obs_dim] initial states
        """
        joined_pred = self.model.decode(latent,attention_mask)
        if self.symlog:
            joined_pred[:, :, :-1] = symexp(joined_pred[:, :, :-1])
        joined_pred[:, :, -1] = torch.sigmoid(joined_pred[:, :, -1])
    
        latent_mask = self.latent_pooling(attention_mask.unsqueeze(1).to(latent.dtype)).squeeze(1)
        joined_pred = joined_pred.masked_fill(latent_mask==0,-100).tolist()
        #only keep the no -100 part
        joined_pred = [x[:x.index(-100)] for x in joined_pred]
        return joined_pred

    def decode_from_indices(self, indices, state):
        """
        Decode a trajectory from latent codes
        Args:
            indices: [B x T] latent codes
            state: [B x obs_dim] or [1 x obs_dim] initial state. [1 x obs_dim] is a shortcut for planning
        """
        B, T = indices.shape
        if self.model.ma_update:
            latent = torch.index_select(self.model.codebook.embedding, dim=0, index=indices.flatten()).reshape([B, T, -1])
        else:
            latent = torch.index_select(self.model.codebook.embedding.weight, dim=0, index=indices.flatten()).reshape(
                [B, T, -1])
        state = state[:,None,:]
        if state.shape[0] == 1 and B > 1:
            state = state.repeat(B, 1, 1)
        joined_pred = self.decode(latent.reshape([B, T//self.code_per_step, -1]), state)
        return joined_pred

    def forward(self, input_ids,attention_mask, labels,question=None):
        """
        Run the full autoencoder model on the given inputs and get loss.
        Args:
            joined_inputs : [ B x T x joined_dimension]
            mask : [ B x T x 1]
            progress: used for discriminator training (optional)
        """

        joined_inputs=self.embed_tokens(input_ids)
        b, t, joined_dimension = joined_inputs.size()
        
        latent_mask = self.latent_pooling(attention_mask.unsqueeze(1).to(joined_inputs.dtype)).transpose(1, 2)

        

        
        
     
        ## [ B x T x embedding_dim ]

        reconstructed_logits, latents, feature= self.model(joined_inputs,attention_mask)

        feature_masked=feature*latent_mask
        latents_masked=latents*latent_mask
        
        
        # VQ objective
        loss_vq = F.mse_loss(latents, feature_masked.detach())
        # Commitment objective
        loss_commit = F.mse_loss(feature_masked, latents.detach())

        

        loss_fn = nn.CrossEntropyLoss()
        reconstructed=reconstructed_logits
        logits=reconstructed_logits.view(-1, self.config.vocab_size)
        labels=labels.view(-1)


        
        reconstruction_loss = loss_fn(logits, labels)


        

        
        return CausalVQVAEOutputWithPast(
            reconstructed=reconstructed,
            feature_masked=feature_masked,
            reconstruction_loss=reconstruction_loss,
            loss_vq=loss_vq,
            loss_commit=loss_commit,
        )
    
    

