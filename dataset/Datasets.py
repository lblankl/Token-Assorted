from transformers import PretrainedConfig,AutoTokenizer
import torch

IGNORE_INDEX = -100

class Filter():
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
        
    


    def __call__(self,examples):

        
        question=examples["query"]
        response=examples["response"]
        txt=question+response
        tokenized_txt=self.tokenizer.encode(txt,add_special_tokens=False)
        examples['length']=len(tokenized_txt)
       
        
        
        
        return examples
class Preprocess():
    def __init__(self,tokenizer,chuck_size=16,max_length=2048):
        self.tokenizer=tokenizer
        self.chuck_size=chuck_size
        self.max_length=max_length
       
    
        
        


    def __call__(self,examples):
        
       
        question=examples["query"]
        response=examples["response"]

        
        
        prompt = self.tokenizer.encode(question,add_special_tokens=True)
        
        answer = self.tokenizer.encode(response,add_special_tokens=False)
       
        len_of_prompt=len(prompt)
        
       
        
        
        sentence=prompt+answer
        #truncate the sentence to max_length
        sentence=sentence[:self.max_length]
        len_of_sentence=len(sentence)
        chunck_num=len_of_sentence//self.chuck_size
        valid_length=chunck_num*self.chuck_size
        sentence=sentence[:valid_length]

        
        exa={}
        exa["input_ids"]=sentence
        
        exa["labels"]=exa["input_ids"].copy()
        exa["question"]=question
        
        return exa
    
from transformers import DataCollatorWithPadding


class CustomCollate:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        #padding="max_length",max_length=1024)

    def __call__(self, examples):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        

        input_ids=[e["input_ids"] for e in examples]
        labels=[e["labels"] for e in examples]
        question=[e["question"] for e in examples]

        
        
        
        padded_input_ids=self.data_collator({"input_ids":input_ids})["input_ids"]
        attention_mask=self.data_collator({"input_ids":input_ids})["attention_mask"]

        padded_labels=self.data_collator({"input_ids":labels})["input_ids"]
        labels_attention_mask=self.data_collator({"input_ids":labels})["attention_mask"]
        
       

        labels_attention_mask=labels_attention_mask.to(torch.bool)
        
        padded_labels=padded_labels.masked_fill(~labels_attention_mask, IGNORE_INDEX)
        
        return {
            "input_ids":padded_input_ids,
            "attention_mask":attention_mask,
            "labels":padded_labels

        }