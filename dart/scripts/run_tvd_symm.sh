
export KL_METHOD='tvd_symm'

python ../kd.py \
  --teacher $SLURM_TMPDIR/models/\
  --data_dir '' \
  --tokenizer_name '' \
  --learning_rate=1e-4 \
  --do_train \
  --gpus 1\
  --sample_beams 6 --do_sample --top_k 30\
  --temperature 1.\
  --val_check_interval 0.3 --n_val -1 --eval_beams 5 --length_penalty=1. \
  --model_name_or_path IGNORED\
  --student '' \
  --alpha_hid=0. --alpha_ce=1. --alpha_mlm=0.\
  --max_source_length 128 --max_target_length=256 --val_max_target_length=256 --test_max_target_length=256 \
  --train_batch_size=8 --eval_batch_size=5 --gradient_accumulation_steps=1 \
  --warmup_steps 100 \
  --output_dir '' \
  --overwrite_output_dir\
  --kd_method $KL_METHOD\
  --num_train_epochs 12\
  "$@"
