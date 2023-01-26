#!/bin/sh


python /home/fdill/chat/distillation.py \
  --teacher /home/fdill/chat/models/gpt2_teacher/best_tfmr/\
  --data_dir /home/fdill/chat/Commonsense-Dialogues/\
  --tokenizer_name $SLURM_TMPDIR/models/ \
  --freeze_embeds \
  --student_layers 4 \
  --learning_rate=5e-4\
  --alpha_hid=0.1 --normalize_hidden\
  --do_train \
  --gpus 1\
  --val_check_interval 0.5 --n_val -1 --eval_beams 1 --length_penalty=1. \
  --model_name_or_path IGNORED \
  --train_batch_size=8 --eval_batch_size=8 --gradient_accumulation_steps=3 \
  --warmup_steps 100 \
  --output_dir $SLURM_TMPDIR/student_models/ \
  --overwrite_output_dir\
  --num_train_epochs 28\