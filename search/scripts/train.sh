#!/bin/bash
echo "begin training"
python search/code/run.py \
    --output_dir=./saved_search \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=codebert \
    --tokenizer_name=roberta-base \
    --do_train \
    --py_train_data_file=dataset/python/train \
    --py_eval_data_file=dataset/python/valid \
    --py_test_data_file=dataset/python/test \
    --js_train_data_file=dataset/javascript/train \
    --js_eval_data_file=dataset/javascript/valid \
    --js_test_data_file=dataset/javascript/test \
    --epoch 10 \
    --block_size 300 \
    --train_batch_size 70 \
    --eval_batch_size 70\
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train.log