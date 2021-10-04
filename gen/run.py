import pickle
import math
import os
import pprint
import sys
import transformers

from tqdm import tqdm
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from MainDataset import MainDataset
from search.code.add_search import ExCode

def run_training(args, train_data,eval_data):

    model=transformers.GPTNeoForCausalLM.from_pretrained("saved_models/checkpoint/") 
    train_data.start_iteration = 0
    training_args = transformers.TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        do_eval=False,
        do_predict=True,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
    
        logging_dir=args.save_dir, 
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_steps=args.save_freq,
        save_total_limit=3,

        dataloader_drop_last=True,
        dataloader_num_workers=1,

        local_rank=args.local_rank,

        deepspeed=args.deepspeed,
        fp16=args.fp16,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data
    )
    trainer.train() 
    model.save_pretrained(os.path.join(args.save_dir, "final_checkpoint"))


def get_dataset(args): 
    #remove this
    #TODO: max tokens? play with it.
    train_data = MainDataset(
        max_input_len=550,
        max_target_len=300,
        max_context_len=250,
        mode='train'
    )
    eval_data = MainDataset(
        max_input_len=550,
        max_target_len=300,
        max_context_len=250,
        mode='valid'
    )
    torch.save(train_data,'data/train.pt')
    torch.save(eval_data,'data/valid.pt')
    
    print('saved tensors')
    pickle.dump(train_data,open('data/train.bin','wb'))
    pickle.dump(eval_data,open('data/valid.bin','wb'))
    # train_data=None
    return train_data,eval_data

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Language Modelling on Code")
    parser.add_argument('--arch', default='gpt2')
    parser.add_argument('--dummy-model', action='store_true')
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--load_train_dataset', default='data/train.bin', type=str)
    parser.add_argument('--load_eval_dataset', default='data/valid.bin', type=str)
    parser.add_argument('--resume', default=None, type=str)
    # Dataloading
    parser.add_argument('--context-dataroot', default='dataset/context_js/', type=str)
    # Training
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    # parser.add_argument('--lr-warmup-steps', default=500, type=int)
    parser.add_argument('--batch-size-per-replica', default=3, type=int)
    parser.add_argument('--grad-acc-steps', default=2, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')
    # Logging and stuff
    parser.add_argument('--save-dir', default="saved_gen/", type=str)
    parser.add_argument('--log_freq', default=1000, type=int)
    parser.add_argument('--save-freq', default=10000, type=int)
    parser.add_argument('--eval_steps', default=5000, type=int)

    args = parser.parse_args()

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    os.makedirs(args.save_dir, exist_ok=True)
    if os.path.exists(args.load_train_dataset) and os.path.exists(args.load_eval_dataset):
        eval_data=pickle.load(open(args.load_eval_dataset,'rb'))
        train_data=pickle.load(open(args.load_train_dataset,'rb'))
        print('original train len:  ',len(train_data))
    else:
        train_data,eval_data = get_dataset(args)

    run_training(args, train_data,eval_data=eval_data)
    
