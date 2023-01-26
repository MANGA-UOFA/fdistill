#!/usr/bin/env bash
export MODEL_NAME=""
export REF_PATH=""
echo $MODEL_NAME
echo $REF_PATH
export BEAM=5
python3 /home/fdill/t5mt/run_calc_entropy.py \
  --model_name $MODEL_NAME \
  --input_path /home/fdill/t5mt/data_sm/test.source \
  --reference_path $REF_PATH \
  --device cuda