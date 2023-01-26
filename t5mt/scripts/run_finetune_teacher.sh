#!/bin/sh


export MODEL_OUTPUT_PATH='/'

python3 /home/fdill/t5mt/finetune.py \
    --learning_rate=5e-4\
    --do_train \
    --val_check_interval=0.5 \
    --adafactor \
    --num_train_epochs 9 \
    --data_dir /home/fdill/t5mt/data_sm/ \
    --max_source_length 300 --max_target_length 300 --val_max_target_length 300 --test_max_target_length 300 \
    --train_batch_size=8 --eval_batch_size=4 --eval_beams 2\
    --n_val -1\
    --seed 42\
    --task translation \
    --warmup_steps 500 \
    --gpus 1\
    --output_dir $MODEL_OUTPUT_PATH \
    --model_name_or_path /home/fdill/t5mt/models/t5-base/ \
    --tokenizer_name /home/fdill/t5mt/models/t5-base/ \
    --overwrite_output_dir \
    "$@"
