#!/bin/sh

export MODEL_OUTPUT_PATH="/"

python /home/fdill/t5mt/distillation.py \
  --teacher /home/fdill/t5mt/models/teacher_en-ro_epc9/best_tfmr/ \
  --num_train_epochs 28\
  --adafactor \
  --data_dir /home/fdill/t5mt/data_sm/ \
  --tokenizer_name /home/fdill/t5mt/models/teacher_en-ro_epc9/best_tfmr/ \
  --student_decoder_layers 1 --student_encoder_layers 3 \
  --learning_rate=1e-3 \
  --freeze_embeds \
  --temperature 2. \
  --do_train \
  --task translation\
  --gpus 1\
  --val_check_interval 0.3 --n_val -1 --eval_beams 4 --length_penalty=1. \
  --model_name_or_path IGNORED --normalize_hidden\
  --alpha_hid=3.\
  --train_batch_size=8 --eval_batch_size=8 --gradient_accumulation_steps=1 \
  --warmup_steps 500\
  --output_dir $MODEL_OUTPUT_PATH \
  --overwrite_output_dir\
  "$@"
