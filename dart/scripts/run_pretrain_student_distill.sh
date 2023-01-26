python ../distillation.py \
  --teacher '' \
  --data_dir '' \
  --tokenizer_name '' \
  --student_decoder_layers 1 --student_encoder_layers 3 \
  --learning_rate=1e-4 \
  --freeze_embeds \
  --do_train \
  --gpus 1\
  --val_check_interval 0.3 --n_val -1 --eval_beams 5 --length_penalty=1. \
  --max_source_length=128 --max_target_length=128 --val_max_target_length=128 --test_max_target_length=128 \
  --model_name_or_path IGNORED \
  --alpha_hid=3. \
  --train_batch_size=8 --eval_batch_size=8 --gradient_accumulation_steps=1 \
  --num_train_epochs=18 \
  --warmup_steps 100\
  --output_dir '' \
  --overwrite_output_dir\
  "$@"
