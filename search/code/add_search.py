from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
import sys 
from search.code.model import Model
import torch
import scann
import pickle,json
import time
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm 
import os

class InputFeatures(object):
	"""A single training/test features for a example."""
	def __init__(self,
					docstring_ids,
					idx,
					docstring,source,code,orig_doc

	):
		self.docstring_ids=docstring_ids
		self.idx=idx
		self.docstring=docstring
		self.source=source
		self.code=code
		self.orig_doc=orig_doc

def convert_examples_to_features(js,tokenizer,block_size):
	#choose the language
	docstring='Language: Javascript'+' NL: '+js['doc']
	orig_doc=js['doc']
	code=js['body']

	docstring_tokens=[tokenizer.cls_token]+tokenizer.tokenize(docstring)[:block_size-2]+[tokenizer.sep_token]
	
	docstring_ids =  tokenizer.convert_tokens_to_ids(docstring_tokens)
	padding_length =block_size - len(docstring_ids)
	docstring_ids+=[tokenizer.pad_token_id]*padding_length

	return InputFeatures(docstring_ids=docstring_ids,idx=js['idx'],docstring=docstring,source=js['source'],code=code,orig_doc=orig_doc)
class TextDataset(Dataset):
	def __init__(self, tokenizer, block_size=100, file_path=None):
		self.examples = []
		data=[]
		f=pickle.load(open(file_path,'rb'))
		for i,line in enumerate(f):
			js=line
			data.append(js)
		for js in data:
			converted_ex=convert_examples_to_features(js,tokenizer,block_size=block_size)
			if converted_ex:
				self.examples.append(converted_ex)
        
	def __len__(self):
		return len(self.examples)

	def __getitem__(self, i):   
		return (torch.LongTensor(self.examples[i].docstring_ids),self.examples[i].idx,self.examples[i].docstring,self.examples[i].source,self.examples[i].code,self.examples[i].orig_doc)
class ExCode(object):
    """example of retrived code with corresponding nl and idx"""
    def __init__(self,codes,idx,nl,source):
        self.codes=codes
        self.idx=idx
        self.nl=nl
        self.source=source
def save_excodes(loader,model,device,searcher,codebase,mode,source_data):
	excodes={}
	
	for step, batch in tqdm(enumerate(loader),total=len(loader)):
		docstring_ids=batch[0]
		idxs=batch[1]
		docstrings=batch[2]
		sources=batch[3]
		src_codes=batch[4]
		orig_docs=batch[5]
	
		with torch.no_grad():
			embeds=model.get_representation_batch(qc_ids=docstring_ids,device=device)
			embeds=embeds.detach().cpu().numpy()
	
		assert len(idxs)==len(embeds)==len(docstrings)==len(sources)==len(src_codes)
		for idx, embed,docstring,source,src_code,orig_doc in zip(idxs,embeds,docstrings,sources,src_codes,orig_docs):
			assert source==source_data
			idx=idx.item()
			code_idx,_=searcher.search(embed)
			filtered_code_idx=[]
			for c_id in code_idx: 
				
				if codebase[c_id]['docstring'].strip() == orig_doc.strip() or codebase[c_id]['language']!='javascript':
					continue 
				else:
					set_c_id_code=set(codebase[c_id]['code'].split(' '))	
					set_src_code=set(src_code.split(' '))
					common=set_c_id_code & set_src_code
					max_per=max(len(common)/len(set_c_id_code),len(common)/len(set_src_code))
					if max_per>=0.95:
						continue 
					else:
						filtered_code_idx.append(c_id)

			codes=[codebase[c_id]['code'] for c_id in filtered_code_idx if codebase[c_id]['language']=='javascript']
			
			excodes[idx]=ExCode(idx=idx,source=source,codes=codes,nl=docstring)
	pickle.dump(excodes,open(f'data/{mode}_{source_data}_excodes.bin','wb'))
	print('done')
	
def main(model,tokenizer,codebase):
	searcher = scann.scann_ops_pybind.load_searcher('data/scann_searcher')
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print('the device is : ',device)
	model.to(device)
	model.eval()
	
	context_train_dataset=TextDataset(tokenizer,file_path='dataset/context_js/train.bin')
	context_valid_dataset=TextDataset(tokenizer,file_path='dataset/context_js/valid.bin')
	print('lengths: ',len(context_train_dataset),len(context_valid_dataset))
	context_train_dataloader = DataLoader(context_train_dataset,batch_size=30)
	context_valid_dataloader = DataLoader(context_valid_dataset,batch_size=30)
	save_excodes(context_train_dataloader,model,device,searcher,codebase,'train',source_data='context')
	save_excodes(context_valid_dataloader,model,device,searcher,codebase,'valid',source_data='context')
	return

if __name__=='__main__':
	model_name='saved_search/checkpoint-best-mrr/model.bin'
	config = RobertaConfig.from_pretrained('codebert',
                                          cache_dir= None)
	tokenizer = RobertaTokenizer.from_pretrained('codebert',
                                                cache_dir=None)
	model = RobertaModel.from_pretrained('codebert',
                                            config=config,
                                           cache_dir=None) 

	codebase= pickle.load(open('data/codebase.bin','rb'))

	model=Model(model,config,tokenizer,args=None)
	model.load_state_dict(torch.load(model_name))
	main(model,tokenizer,codebase)
	

