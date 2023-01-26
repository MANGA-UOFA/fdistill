#!/bin/sh

export KL_METHOD="seqkd"
export MODEL_SAVE_PATH="/"
export INIT_MODEL='/'
export DATA_DIR='/'
export TEACHER_MODEL='/'


CUDA_LUANCH_BLOCKING=1 python /home/fdill/t5mt/kd.py \
  --teacher $TEACHER_MODEL \
  --data_dir $DATA_DIR\
  --adafactor \
  --tokenizer_name $INIT_MODEL\
  --learning_rate=5e-4 \
  --do_train \
  --gpus 1\
  --task translation\
  --temperature 1.\
  --freeze_embeds \
  --val_check_interval 0.5 --n_val -1 --eval_beams 5 --length_penalty=1. \
  --model_name_or_path IGNORED\
  --student $INIT_MODEL \
  --alpha_hid=0. --alpha_ce=1. --alpha_mlm=0.\
  --train_batch_size=8 --eval_batch_size=5 --gradient_accumulation_steps=1 \
  --warmup_steps 100 \
  --output_dir $SLURM_TMPDIR/student_models/ \
  --overwrite_output_dir\
  --kd_method $KL_METHOD\
  --num_train_epochs 12\
  "$@"
