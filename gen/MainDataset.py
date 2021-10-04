import torch
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import numpy as np
import os
import io, pickle
from search.code.add_search import ExCode
import transformers
import io, tokenize
from tqdm import tqdm 
from docstring_parser import parse
import json

#adapted from: https://stackoverflow.com/a/62074206
def remove_comments_and_docstrings(source):
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = '\n'.join(l for l in out.splitlines() if l.strip())
    return out

class Example(object):
    def __init__(self,
                 idx,
                 nl,
                 code,
                 source,
                 context
                 ):
        self.idx = idx
        self.nl = nl
        self.code = code
        self.source=source
        self.context=context



class MainDataset(torch.utils.data.Dataset):
    def __init__(self, max_input_len,max_target_len,max_context_len,mode=None):
        self.max_input_len=max_input_len
        self.max_target_len=max_target_len
        self.max_context_len=max_context_len
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
        self.mode=mode
        self.inp_labels=[]
        self.initialize()
        

    def read_examples(self,filename,source):
        examples=[]
        f=pickle.load(open(filename,'rb'))
        for i, line in enumerate(f):
            js=line
            if 'idx' not in js or 'source' not in js:
                print("doesn't have any idx")
                exit()
            nl=js['doc'].replace('\n','')
            code=js['body']
            context=js['context']
            examples.append(
                Example(
                        idx = js['idx'],
                        nl=nl,
                        code = code,
                        source=source,context=context
                        ) 
            )
        return examples    
    def initialize(self):
        context_examples=self.read_examples(f'dataset/context_js/{self.mode}.bin',source='context')
        examples=context_examples
        np.random.seed(69)
        np.random.shuffle(examples)
        if self.mode=='valid':
            self.choose_rand_ex=np.random.choice([0,1],len(examples),p=[0,1])
            self.choose_rand_context=np.random.choice([0,1],len(examples),p=[0,1])
        else:
            self.choose_rand_ex=np.random.choice([0,1],len(examples),p=[0.45,0.55])
            self.choose_rand_context=np.random.choice([0,1],len(examples),p=[0.40,0.60])

        context_excodes=pickle.load(open(f'data/{self.mode}_context_excodes.bin','rb'))
        inp_labels=[]
     
        for i,ex in enumerate(examples):
            query,code,idx,src,context=ex.nl, ex.code,ex.idx,ex.source,ex.context
            
            excode=context_excodes[idx]
            if not context:
                context_tokens='None\n'
            else:
                context_tokens=context
                context_tokens+='\n'

            query_tokens="Query:\n"+query+'\n'
            query_ids=self.tokenizer.encode(query_tokens)

            if self.choose_rand_context[i]:
                context_ids=self.tokenizer.encode('Context:\n')+self.tokenizer.encode(context_tokens)[-self.max_context_len:]
            else:
                context_ids=self.tokenizer.encode('Context:\n')+self.tokenizer.encode('None\n')[-self.max_context_len:]
            excode_tokens='Examples from search:\n'
            if self.choose_rand_ex[i]:
                for i_e_code,e_code in enumerate(excode.codes):
                
                    if i_e_code>1:break
                    excode_tokens+=e_code
                    excode_tokens+='\n'
            else:
                excode_tokens+='None\n'
            
            excode_ids=self.tokenizer.encode(excode_tokens)
            total_query_ids=context_ids+query_ids+excode_ids
            add_newline=False
            if len(total_query_ids)>self.max_input_len:add_newline=True
            total_query_ids=total_query_ids[:self.max_input_len]
            if not add_newline: 
                code_prompt='Generate Code:\n'+code
            else: 
                code_prompt='\nGenerate Code:\n'+code
            code_ids=self.tokenizer.encode(code_prompt,verbose=False)[:self.max_target_len-1]
            code_ids.append(self.tokenizer.eos_token_id)
            input_ids=total_query_ids+code_ids
            labels=[-100]*len(total_query_ids)+code_ids

            #add paddings
            padding_length=self.max_input_len+self.max_target_len-len(input_ids)
            input_ids+=[self.tokenizer.eos_token_id]*padding_length
            labels+=[-100]*padding_length
            inp_labels.append({
                'input_ids':torch.LongTensor(input_ids),
                'labels':torch.LongTensor(labels)
            })
            
        self.inp_labels=inp_labels
        
    def __len__(self):
        return len(self.inp_labels)


    def __getitem__(self, idx):
        return self.inp_labels[idx]

