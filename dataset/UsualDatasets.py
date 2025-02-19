
from transformers import PretrainedConfig,AutoTokenizer
import torch

IGNORE_INDEX = -100
class Filter():
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
        
    


    def __call__(self,examples):

        format_str="The answer is: "
        question=examples["query"]
        response=examples["response"]
        think=response.split(format_str)[0]
        txt=question+response
        tokenized_txt=self.tokenizer.encode(txt,add_special_tokens=False)
        examples['length']=len(tokenized_txt)
        examples['think']=len(think)
        return examples

class Preprocess():
    def __init__(self,tokenizer,prompt_template=None,response_template=None,max_length=2048,standard=False):
        self.tokenizer=tokenizer
        self.standard=standard
        if prompt_template is not None:
            self.prompt_template=prompt_template
        else:
            self.prompt_template="""The user asks a question, and the Assistant solves it.
            The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. 
            The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
            \n\nUser:{prompt}\nAssistant: """
        if response_template is not None:
            self.response_template=response_template
        else:
            self.response_template="""<think> {think} </think><answer> {answer} </answer>"""
    
        
        


    def __call__(self,examples):
        
       
        question=examples["query"]
        response=examples["response"]

        format_str="The answer is: "
        if self.standard:
            think=examples["think"]
            answer=examples["answer"]
        else:
            think=response.split(format_str)[0]
            answer=response.split(format_str)[1]

        question = self.prompt_template.format(prompt=question)
        response = self.response_template.format(think=think, answer=answer)
        
        prompt = self.tokenizer.encode(question,add_special_tokens=True)
        
        answer = self.tokenizer.encode(response,add_special_tokens=False)+[self.tokenizer.eos_token_id]
       
        len_of_prompt=len(prompt)
        
       
        
        
        sentence=prompt+answer
        exa={}
        exa["input_ids"]=sentence
        
        exa["labels"]=exa["input_ids"].copy()
        
        
        
        
        #set the prompt token to -100   
        exa["labels"][:len_of_prompt]=[IGNORE_INDEX]*len_of_prompt
        
        
        
        
        return exa
    
from transformers import DataCollatorWithPadding


class CustomCollate:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        #padding="max_length",max_length=1024)

    def __call__(self, examples):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        

        input_ids=[e["input_ids"] for e in examples]
        labels=[e["labels"] for e in examples]
        
        
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