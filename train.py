
from accelerate import Accelerator

import os 
import torch.nn as nn

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from transformers import (
    HfArgumentParser

)
import numpy
import random
import mlflow
import collections
from dataclasses import dataclass, field
from typing import Optional

import os

from accelerate.utils import tqdm
import torch  # noqa

from dataset.Datasets import CustomCollate,Preprocess,Filter
from transformers import (
    AutoTokenizer,
    AutoConfig,
)
import wandb
from datasets import concatenate_datasets
from datasets import load_from_disk
from peft import LoraConfig,PeftModel
from peft import get_peft_model

from transformers import get_scheduler,AutoModelForCausalLM

from model.vqvae import VQVAE
@dataclass
class ModelArguments:
    embedding_size: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "embedding size."
            )
        }
    )
    hidden_size: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "hidden size."
            )
        }
    )
    K: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "codeblock length."
            )
        }
    )
    latent_step: Optional[int] = field(
        default=16,
        metadata={
            "help": (
                "latent step.Also known as block size of assorted tokens"
            )
        }
    )

    num_layers: Optional[int] = field(
        default=2,
        metadata={
            "help": (
                "The number of layers to use."
            )
        }
    )

    base_model_type: Optional[str] = field(
        default="llama",
        metadata={
            "help": (
                "The model type to use."
            )
        }
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization."
            )
        }
    )

    tokenizer: str = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    
    
    

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    load_disk: Optional[bool] = field(
        default=False, metadata={"help": "Whether load from disk"}
    )

    data_range: Optional[int] = field(
        default=None, metadata={"help": "data range"}
    )
   
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
   
    data_point: Optional[int] = field(
        default=0, metadata={"help": "ckpt data point"}
    )
    

@dataclass
class HiddenTrainingArguments:
    beta: Optional[float] = field(
        default=1.1, metadata={"help": "The beta"}
    )
    warmup_steps: Optional[int] = field(
        default=0, metadata={"help": "The number of warmup steps"}
    )
    learning_rate: Optional[float] = field(
        default=2e-5, metadata={"help": "The initial learning rate for AdamW"}
    )
    
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether use gradient checkpointing"}
    )
    print_div: Optional[int] = field(
        default=100, metadata={"help": "The print div"}
    )
   
    attention_implement: str = field(
        default="eager", metadata={"help": "attention implement"}
    )
        
   
    debugging: bool = field(
        default=False, metadata={"help": "debug mode"}
    )
    name: Optional[str] = field(
        default=None, metadata={"help": "The name of logging"}
    )
    logging_dir:  str= field(
        default=None, metadata={"help": "logging base directory"}
    )
    output_dir:  str= field(                                                                    
        default=None, metadata={"help": "out put directory"}
    )
   
    per_device_train_batch_size: Optional[int] = field(
        default=1, metadata={"help": "The batch size for training"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "The batch size for evaluation"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=32, metadata={"help": "The gradient accumulation steps"}
    )
    evaluation_strategy: Optional[str] = field(
        default="steps", metadata={"help": "The evaluation strategy"}
    )
    eval_steps: Optional[int] = field(
        default=int(500e3), metadata={"help": "The evaluation steps"}
    )#64
    logging_steps: Optional[int] = field(
        default=32, metadata={"help": "The logging steps"}
    )
    save_steps: Optional[int] = field(
        default=6000, metadata={"help": "The saving steps"}
    )
    save_total_limit: Optional[int] = field(
        default=10, metadata={"help": "The saving total limit"}
    )
    bf16: Optional[bool] = field(
        default=True, metadata={"help": "Whether use bf16"}
    )
    dataloader_num_workers: Optional[int] = field(
        default=2, metadata={"help": "The dataloader num workers"}
    )
  
    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "The number of training epochs"}
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "The seed"}
    )
    checkpoint: Optional[int] = field(
        default=None, metadata={"help": "The checkpoint"}
    )
 
    grad_clip: Optional[float] = field(
        default=5.0, metadata={"help": "The checkpoint"}
    )
   
    ckpt: str = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
def main():
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, HiddenTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # wandb.login()
    wandb.init(
      # Set the project where this run will be logged
      project="ShortR1",
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"experiment_{training_args.name}",
      # Track hyperparameters and run metadata
      config={
      "learning_rate": training_args.learning_rate,
      "architecture": model_args.base_model_type,
      "dataset": data_args.dataset_name,
      "epochs": training_args.num_train_epochs,
      })
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        numpy.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    setup_seed(training_args.seed)

    #datasets
    if data_args.load_disk:
        train_datasets = load_from_disk(data_args.dataset_name)
    else:
        train_datasets = load_dataset(data_args.dataset_name,split="train",trust_remote_code=True)
    

    
    train_datasets = train_datasets.shuffle(42)
    data_len=train_datasets.shape[0]
    
    if data_args.data_range is not None:
        data_len=data_args.data_range
        train_datasets = train_datasets.select(range(data_args.data_point,data_args.data_range))

   
   

    

    tokenizer=AutoTokenizer.from_pretrained(model_args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
   


    from torch.utils.data import DataLoader

    ##config
    
   
   
   
    cfg=AutoConfig.from_pretrained(model_args.model_name_or_path)
    cfg.embedding_size=model_args.embedding_size
    cfg.base_hidden_size=cfg.hidden_size
    cfg.hidden_size = model_args.hidden_size
    cfg._attn_implementation=training_args.attention_implement
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 4
    cfg.K = model_args.K
    cfg.latent_step = model_args.latent_step
    cfg.base_model_type = model_args.base_model_type
    cfg.intermediate_size=cfg.hidden_size
    # cfg.hidden_n=training_args.hidden_n
    
    if training_args.debugging==True:
            
            
        
        
        cfg.hidden_size = 512
        cfg.num_hidden_layers = 2
        cfg.num_attention_heads = 12
        cfg.num_key_value_heads = 3
        cfg.intermediate_size = 256
        cfg.vocab_size = 128256
        cfg.num_infer_layers=2
        # cfg.hidden_n=training_args.hidden_n
        cfg._attn_implementation=training_args.attention_implement

    
        model = VQVAE(cfg)

    
    else:
        model = VQVAE(cfg)
  
    if cfg.embedding_size==cfg.base_hidden_size:
        print("initlize the embedding of model with the embedding of reference model")
        Reference_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

        with torch.no_grad():
            #initlize the embedding of model with the embedding of reference model
            model.embed_tokens.weight.data=Reference_model.model.embed_tokens.weight.data
        del Reference_model
            



    dfilter=Filter(tokenizer)
    
    train_datasets=train_datasets.map(dfilter)
    train_datasets=train_datasets.filter(lambda x: x['length']> model_args.latent_step)

    preprocess=Preprocess(tokenizer,model_args.latent_step)

    train_datasets=train_datasets.map(preprocess)
    
    
    collator=CustomCollate(tokenizer)
    print_div=training_args.print_div
   
        

    output_dir=training_args.output_dir

   

 
    if training_args.gradient_checkpointing:
       
        model.config.use_cache = False         # required for gradient checkpointing
        model.enable_input_require_grads()     # required for gradient checkpointing
        model.gradient_checkpointing_enable()  # enable gradient checkpointing  
    # model.gradient_checkpointing_enable()
    
    
    epochs=training_args.num_train_epochs
    eval_steps=training_args.eval_steps
    batch_size=training_args.per_device_train_batch_size
    eval_batch_size=training_args.per_device_eval_batch_size
    
    logging_dir=training_args.logging_dir
    logging_steps=training_args.logging_steps
    save_steps=training_args.save_steps
    save_total_limit=2
    accumulation_step=training_args.gradient_accumulation_steps
    if training_args.checkpoint is not None:
        checkp=training_args.checkpoint
    else:
        checkp=0

    hps={"epochs":epochs,"batch_size":batch_size,"eval_batch_size":eval_batch_size,
         "output_dir":output_dir,"logging_dir":logging_dir,"eval_steps":eval_steps,
         "logging_steps":logging_steps,"save_steps":save_steps,"save_total_limit":save_total_limit}
    
    import os
    #check if the logging directory exists
    name=training_args.name
    if not os.path.exists(logging_dir+'/'+training_args.name):
        pass
    else:
        # if exists, list all the folders under the logging directory
        dirs = os.listdir(logging_dir+'/'+training_args.name)
        
        #filter all the name that have 'ckpt' in the name, this folder is in this form:  ckpt-n   n is the ckpt time
        dirs = [i for i in dirs if 'ckpt' in i]
        
        if len(dirs)==0:
            maxn=0
        else:
            #find the latest n
            dirs = [int(i.split('-')[1]) for i in dirs]
            dirs.sort()
            maxn=dirs[-1]
        #create a new folder with the name ckpt-maxn+1
        logging_dir = logging_dir + '/'+name
        name='ckpt-' + str(maxn+1)
        
    print("de1")
    accelerator = Accelerator(log_with="tensorboard",project_dir=logging_dir,gradient_accumulation_steps=accumulation_step)
    
    accelerator.init_trackers(name,hps)

    dataloader=DataLoader(train_datasets,batch_size=batch_size,collate_fn=collator,num_workers=training_args.dataloader_num_workers)
    


    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=training_args.learning_rate)

   
    #cosine learning rate scheduler initial lr=2e-5 weight decay=0.1
    data_len=len(train_datasets)
    num_training_steps=int(data_len//(accelerator.num_processes*batch_size*accumulation_step)*epochs)

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=num_training_steps,

    )
  
    train_dataloader,model,optimizer,lr_scheduler = accelerator.prepare(
        dataloader,model,optimizer,lr_scheduler
    )

    skipped_dataloader = None
    if training_args.ckpt:
        accelerator.load_state(training_args.ckpt)
        skipped_dataloader = accelerator.skip_first_batches(train_dataloader, checkp)
    else:
        #check the output directory exists
        if not os.path.exists(output_dir):
            pass
        else:
            #list all the folders under the output directory
            dirs = os.listdir(output_dir)
            if len(dirs)==0:
                pass
            else:
                #find the folder that have largest number  dir name are in the form like state300 
                
                dirs = [int(i.split('state')[1]) for i in dirs if 'state' in i]
                if len(dirs)==0:
                    pass
                else:
                    dirs.sort()
                    ckptnum=dirs[-1]
                    #load the model from the folder
                    accelerator.load_state(output_dir+"/state"+str(ckptnum))
                    skipped_dataloader = accelerator.skip_first_batches(train_dataloader, ckptnum)
                    checkp=ckptnum
            
    if skipped_dataloader:
        dataloader=skipped_dataloader
    else:
        dataloader=train_dataloader
    steps=checkp
    log_steps=checkp//logging_steps
    
    
    state_num=0
    l_acu=0
 
    #reconstruction_loss, loss_vq, loss_commit
    rec_loss_acu=0
    loss_vq_acu=0
    loss_commit_acu=0

    num_epochs=epochs
 
    accelerator.wait_for_everyone()
   

    mlflow.autolog()
    for epoch in tqdm(range(num_epochs), desc="Epoch"):
       
        for batch in tqdm(dataloader):
  
            model.train()
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                outputs=model(**batch)
                #reconstructed, feature_masked, reconstruction_loss, loss_vq, loss_commit
                reconstruction_loss=outputs.reconstruction_loss
                loss_vq=outputs.loss_vq
                loss_commit=outputs.loss_commit
                loss=reconstruction_loss+loss_vq+loss_commit*training_args.beta
                

                
                
                accelerator.backward(loss)
                
                #accelerator.clip_grad_norm_(model.parameters(), training_args.grad_clip)
                optimizer.step()
                lr_scheduler.step()
            
            l_acu+=loss.item()/accumulation_step
            
            rec_loss_acu+=reconstruction_loss.item()/accumulation_step
            loss_vq_acu+=loss_vq.item()/accumulation_step
            loss_commit_acu+=loss_commit.item()/accumulation_step
        

            if steps%logging_steps==0 and steps!=0:
                

                if log_steps%print_div==0:
                    accelerator.print("training_loss",l_acu)
                    accelerator.print("rec_loss_acu",rec_loss_acu)
                    accelerator.print("loss_vq_acu",loss_vq_acu)
                    accelerator.print("loss_commit_acu",loss_commit_acu)

                
                    
                log_steps+=1
                
                
                
           
                if accelerator.is_main_process:
                    
                    mlflow.log_metric("rec_loss_acu",rec_loss_acu,step=log_steps)
                    
                    mlflow.log_metric("loss_vq_acu",loss_vq_acu,step=log_steps)
                    
                    mlflow.log_metric("loss_commit_acu",loss_commit_acu,step=log_steps)
                    
                    mlflow.log_metric("training_loss",l_acu,step=log_steps)
                    
                    wandb.log({"rec_loss_acu":rec_loss_acu,
                               "loss_vq_acu":loss_vq_acu,
                               "loss_commit_acu":loss_commit_acu,
                               "training_loss":l_acu})
                    

                
                

            if steps%accumulation_step==0:
                

                l_acu=0
                rec_loss_acu=0
                loss_vq_acu=0
                loss_commit_acu=0
                
            # if steps%eval_steps==0:
                
            #     model.eval()
            #     with torch.no_grad():
            #         eval+=1
            #         eval_loss=0
            #         temp=0
            #         for batch in test_dataloader:
            #             # if batch["input_for_concept"].shape[1]<=1024:
            #             outputs = model(**batch)
            #             eloss=outputs.loss
                        
            #             #perplexity=torch.exp(loss)
                        
            #             all_loss=accelerator.gather_for_metrics(eloss)
            #             all_loss=all_loss.mean()
            #             eval_loss+=all_loss.item()
            #             temp+=1
            #         eval_loss=eval_loss/temp
            #         accelerator.print("eval_loss:",eval_loss)
            #         accelerator.log({"eval_loss":eval_loss},step=eval)
            if steps%save_steps==0 and steps!=0:
            
                state_num+=1
                
                accelerator.wait_for_everyone()
                
            
                model_state_dict=accelerator.get_state_dict(model)
                unwarp_model=accelerator.unwrap_model(model)
                
                if model_state_dict is not None:
                
                    unwarp_model.save_pretrained(output_dir+"/"+str(steps),is_main_process=accelerator.is_main_process, save_function=accelerator.save,
                                                 state_dict=model_state_dict)


            if steps%(save_steps*1)==0 and steps!=0:
                
                accelerator.save_state(output_dir=training_args.output_dir+"/"+"state"+str(steps))

            steps+=1
    
    accelerator.wait_for_everyone()
    model_state_dict=accelerator.get_state_dict(model)
    unwarp_model=accelerator.unwrap_model(model)
    
    if model_state_dict is not None:
        unwarp_model.save_pretrained(output_dir+"/"+'end',is_main_process=accelerator.is_main_process, save_function=accelerator.save,
                                        state_dict=model_state_dict)
    mlflow.end_run()
    accelerator.end_training()
    accelerator.print("end")

if __name__ == "__main__":
    
   main()