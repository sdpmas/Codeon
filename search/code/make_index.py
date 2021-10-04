import numpy as np
from torch.utils import data
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from model import Model
import torch
import pickle
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
import torch.nn.functional as f
from tqdm import tqdm 
import scann

class InputFeatures(object):
	"""A single training/test features for a example."""
	def __init__(self,
					code_tokens,
					code_ids,
					original_code,
					docstring,url,language

	):
		self.code_tokens = code_tokens
		self.original_code=original_code
		self.code_ids = code_ids
		self.docstring=docstring
		self.url=url
		self.language=language

def convert_examples_to_features(js,tokenizer,block_size):
	#code
	docstring=' '.join(js['docstring_tokens']).strip()
	code='Code: '+' '.join(js['function_tokens'])
	code_tokens=tokenizer.tokenize(code)[:block_size-2]
	code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
	code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
	padding_length =block_size - len(code_ids)
	code_ids+=[tokenizer.pad_token_id]*padding_length
	# print('js: ',js)
	
	original_code=js['function']

	return InputFeatures(code_tokens,code_ids,original_code=original_code,docstring=docstring,url=js['url'],language=js['language'])
class TextDataset(Dataset):
	def __init__(self, tokenizer, block_size=256, file_paths=None):
		self.examples = []
		data=[]
		f_py=pickle.load(open(file_paths['py'],'rb'))
		f_js=pickle.load(open(file_paths['js'],'rb'))
		for i,line in enumerate(f_py):
			js=line
			data.append(js)
		print('len of py: ',len(data))
		for i,line in enumerate(f_js):
			js=line
			data.append(js)
		print('len of js+py: ',len(data))
		np.random.seed(69)
		np.random.shuffle(data)
		for js in data:
			self.examples.append(convert_examples_to_features(js,tokenizer,block_size=block_size))
        
	def __len__(self):
		return len(self.examples)

	def __getitem__(self, i):   
		return (torch.tensor(self.examples[i].code_ids),self.examples[i].original_code,self.examples[i].docstring,self.examples[i].url,self.examples[i].language)

def main(model,tokenizer,js_dataset_file,py_dataset_file):
	# show_gpu('GPU memory usage initially:')
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print('the device is : ',device)
	model.to(device)
	model.eval()
	# model=model.cuda()
	dataset=TextDataset(tokenizer,file_paths={'js':js_dataset_file,'py':py_dataset_file})
	sampler = SequentialSampler(dataset) 
    
	dataloader = DataLoader(dataset, sampler=sampler, 
									batch_size=15)

	codebase={}
	idx_count=0
	reprs=[]
	for step, batch in tqdm(enumerate(dataloader),total=len(dataloader)):
		codes=batch[0]
		orig=batch[1]
		docstrings=batch[2]
		urls=batch[3]
		languages=batch[4]
		
		with torch.no_grad():
			embeds=model.get_representation_batch(qc_ids=codes,device=device)
			embeds=embeds.cpu()
			
		assert len(embeds)==len(orig)==len(docstrings)

		for embed,orig_code,docstring,url,language in zip(embeds,orig,docstrings,urls,languages):
			codebase[idx_count]={'code':orig_code,'docstring':docstring,'url':url,'language':language}
			reprs.append(embed)
			idx_count+=1
	scann_dataset=torch.stack(reprs)
	normalized_dataset=f.normalize(scann_dataset)
	searcher = scann.scann_ops_pybind.builder(normalized_dataset, 10, "dot_product").tree(
		num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
		2, anisotropic_quantization_threshold=0.2).reorder(100).build()
	searcher.serialize('data/scann_searcher')
	print('idx count: ',idx_count)
	print('searcher saved')
	pickle.dump(codebase,open('data/codebase.bin','wb'))
	print('codebase saved')


if __name__=='__main__':
	model_name='saved_search/checkpoint-best-mrr/model.bin'
	config = RobertaConfig.from_pretrained('codebert',
                                          cache_dir= None)
	tokenizer = RobertaTokenizer.from_pretrained('codebert',
                                                cache_dir=None)
	model = RobertaModel.from_pretrained('codebert',
                                            config=config,
                                           cache_dir=None) 
	
	py_dataset_file='dataset/python/py_superset.bin'
	js_dataset_file='dataset/js/js_superset.bin'

	model=Model(model,config,tokenizer,args=None)
	model.load_state_dict(torch.load(model_name))
	main(model,tokenizer,js_dataset_file=js_dataset_file,py_dataset_file=py_dataset_file)