#!/bin/sh

export KD_METHOD='seqkd'
export SAVE_MODEL='/home/fdill/chat/models_freeze/student_4L_seqkd/'
export INIT_MODEL='/home/fdill/chat/models/student_4L_hid/best_tfmr/'
export TEACHER_MODEL='/home/fdill/chat/models/teacher/best_tfmr/'

python /home/fdill/chat/kd.py \
  --teacher $TEACHER_MODEL\
  --data_dir /home/fdill/chat/cd_teacher_label/\
  --tokenizer_name $TEACHER_MODEL \
  --learning_rate=1e-4 \
  --do_train \
  --gpus 1\
  --freeze_embeds \
  --disable_monitor\
  --val_check_interval 0.5 --n_val -1 --eval_beams 1 --length_penalty=1. \
  --model_name_or_path IGNORED\
  --student $INIT_MODEL \
  --train_batch_size=8 --eval_batch_size=5 --gradient_accumulation_steps=2 \
  --warmup_steps 100 \
  --output_dir $SLURM_TMPDIR/student_models/ \
  --overwrite_output_dir\
  --kd_method $KD_METHOD\
  --num_train_epochs 12\
  "$@"

cp $SLURM_TMPDIR/student_models/ $SAVE_MODEL -r