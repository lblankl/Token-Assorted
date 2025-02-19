# 
import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
from peft import LoraConfig
import argparse
from transformers import TrainingArguments,Trainer
from transformers import LlamaTokenizer
from transformers import LlamaConfig
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments,AutoConfig
from transformers import LlamaForCausalLM
from dataset.AssortedDataset import CustomCollate,Preprocess,Filter
import torch
from trainer.testtrainer import TestTrainer 
# import deepspeed
# deepspeed.ops.op_builder.CPUAdamBuilder().load()
# from trainer.ConceptTrainer import CTrainer
import mlflow
from transformers.trainer_utils import get_last_checkpoint
from transformers import AdamW
from peft import LoraConfig
from peft import get_peft_model
IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
from transformers import TrainingArguments,Trainer
from datasets import load_from_disk
from datasets import concatenate_datasets

from model.vqvae import VQVAE
class ModelArguments:
    def __init__(self):
      
        self.logging_dir="/mnt/danlongyuan/ShortR1/records/log/Llama-3.2-3B-Assorted-MetaMath"
        self.tokenizer= "/mnt/danlongyuan/resource/models/huggingface/meta-llama/Llama-3.2-3B"
        self.modelpath = "/mnt/danlongyuan/resource/models/huggingface/meta-llama/Llama-3.2-3B"
        self.output_dir= "/mnt/danlongyuan/ShortR1/records/out/Llama-3.2-3B-Assorted-MetaMath"
        self.dataset_name='/mnt/danlongyuan/resource/dataset/MetaMath'
        self.data_range=150000
        self.gckpt=True
        self.peft=False
        self.attention_implement="flash_attention_2"
        self.checkpoint=None
        self.load_disk=False

        self.K=1024
        self.latent_step=16
        self.vqvae="VQVAE-Llama3.2-3B-MetaMath"
    
def main():

    args=ModelArguments()
    
    mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH=100
    mlflow.utils.validation.MAX_PARAM_VAL_LENGTH=100
    training_args = TrainingArguments(
        deepspeed="deepspeedcfg/ds_config_zero3.json",
                                      output_dir=args.output_dir,
                                      num_train_epochs=1,
                                      overwrite_output_dir=True,per_device_train_batch_size=2,
                                      per_device_eval_batch_size=2,gradient_accumulation_steps=16,
                                      evaluation_strategy="steps",eval_steps=100000,logging_steps=1,
                                      save_steps=300,save_total_limit=10,logging_dir=args.logging_dir,
                                      report_to="mlflow",do_eval=True,gradient_checkpointing=args.gckpt,
                                      learning_rate=5e-6,weight_decay=0.01,warmup_steps=20,adam_epsilon=1e-08,
                                    bf16=True,
                                      dataloader_num_workers=20,remove_unused_columns=False)    
    #datasets
    data_args=args
    if data_args.load_disk:
        train_datasets = load_from_disk(data_args.dataset_name)
    else:
        train_datasets = load_dataset(data_args.dataset_name,split="train",trust_remote_code=True)
 
    train_datasets=train_datasets.shuffle(42).select(range(args.data_range))

    

    d=train_datasets[0]
    dataset=train_datasets
    test_dataset =train_datasets.select(range(4))


    

    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer)
    dfilter=Filter(tokenizer)
    dataset=dataset.map(dfilter)
    dataset.filter(lambda x: x['length']> args.latent_step and x['think']> 0)


    model=AutoModelForCausalLM.from_pretrained(args.modelpath,_attn_implementation=args.attention_implement)

    vqvae=VQVAE.from_pretrained(args.vqvae)
    latent_tempalte="<Latent-code-{number}>"
    for i in range(0, args.K):
        tokenizer.add_special_tokens({"additional_special_tokens": [latent_tempalte.format(number=i)]})
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
            
        model.model.embed_tokens.weight.data[
            -args.K:] = vqvae.model.codebook.weight.data
        

    config=AutoConfig.from_pretrained(args.modelpath,token=token)
    tokenizer.pad_token_id=config.bos_token_id
    
    
    preprocess=Preprocess(tokenizer,vqvae)

    dataset=dataset.map(preprocess)
    test_dataset=test_dataset.map(preprocess)
  

    


    ##########
    
    if args.peft:
        pmodel=get_peft_model(model,LoRAconfig)
        
        pmodel.print_trainable_parameters()
    else:
        pmodel=model
    pmodel=pmodel.half()
    if args.gckpt:
        pmodel.config.use_cache = False         # required for gradient checkpointing
        pmodel.enable_input_require_grads()     # required for gradient checkpointing
        pmodel.gradient_checkpointing_enable()  # enable gradient checkpointing
    trainer = TestTrainer(model=pmodel, args=training_args,train_dataset=dataset,
                       eval_dataset= test_dataset,data_collator=CustomCollate(tokenizer=tokenizer))
    
    # import os
    # if os.path.exists(training_args.output_dir):
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    # else:
    #     last_checkpoint = args.checkpoint
    # checkpoint=last_checkpoint
    # trainer.train(resume_from_checkpoint=checkpoint)
    trainer.train()
    trainer.save_model(output_dir=args.output_dir+'/end')

if __name__ == "__main__":
    
    main()
