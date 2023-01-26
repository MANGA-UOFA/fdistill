#!/bin/sh


CUDA_LAUNCH_BLOCKING=1 python3 /home/fdill/chat/finetune_pair.py \
    --learning_rate=5e-5 \
    --gpus 1 \
    --do_train \
    --n_val -1 \
    --num_train_epochs 9\
    --warmup_steps=100\
    --train_batch_size=8 --eval_batch_size=4 --gradient_accumulation_steps=3 \
    --val_check_interval 0.5 --eval_beams 1\
    --data_dir /home/fdill/chat/Commonsense-Dialogues/\
    --model_name_or_path /home/fdill/chat/models/DialoGPT-medium/\
    --output_dir $MODEL_OUTPUT_PATH \
    --overwrite_output_dir\
    "$@"
