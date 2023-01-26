python3 ../finetune.py \
    --learning_rate=1e-5 \
    --gpus 1 \
    --do_train \
    --do_predict \
    --n_val -1 \
    --num_train_epochs 16 \
    --warmup_steps=100\
    --train_batch_size=8 --eval_batch_size=4 --gradient_accumulation_steps=1 \
    --max_source_length 128 --max_target_length=256 --val_max_target_length=256 --test_max_target_length=256\
    --val_check_interval 0.5 --eval_beams 5\
    --data_dir '' \
    --model_name_or_path '' \
    --output_dir $MODEL_OUTPUT_PATH \
    --overwrite_output_dir\
    "$@"
