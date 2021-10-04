# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args=None):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    def get_representation_batch(self,qc_ids=None,device=None):
        """get represenations in batch for either queries or codes"""
        return self.encoder(qc_ids.to(device),attention_mask=qc_ids.ne(1).to(device))[1]

    def get_representation_one(self,query,device=None):
        """get representation for a single query: less dataset stuffs."""
        query_tokens=[self.tokenizer.cls_token]+self.tokenizer.tokenize(query)[:298]+[self.tokenizer.sep_token]
        query_ids=torch.tensor(self.tokenizer.convert_tokens_to_ids(query_tokens)).unsqueeze(dim=0).to(device)
        return self.encoder(query_ids,attention_mask=query_ids.ne(1))[1].squeeze(dim=0)

    def forward(self, code_inputs,nl_inputs,return_vec=False): 
        bs=code_inputs.shape[0]
        inputs=torch.cat((code_inputs,nl_inputs),0)
        outputs=self.encoder(inputs,attention_mask=inputs.ne(1))[1]
        code_vec=outputs[:bs]
        nl_vec=outputs[bs:]
        
        if return_vec:
            return code_vec,nl_vec
        scores=(nl_vec[:,None,:]*code_vec[None,:,:]).sum(-1)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(bs, device=scores.device))
        return loss,code_vec,nl_vec

      
        
 
