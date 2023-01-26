#!/usr/bin/env bash
export MODEL_NAME='/'
echo $MODEL_NAME
export BEAM=5
python3 /home/lcc/fdill/t5mt/run_eval.py \
  --model_name $MODEL_NAME \
  --input_path /home/fdill/t5mt/data_sm/test.source \
  --save_path $SLURM_TMPDIR/mt.out \
  --reference_path /home/fdill/t5mt/data_sm/test.target \
  --score_path $SLURM_TMPDIR/metrics.json \
  --device cuda\
  "$@"