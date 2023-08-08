#!/bin/sh
export MODEL_NAME='/sracth/dart/models/teacher_ft_lr1e-5_epc16_bsz8/best_tfmr/'
export REF_PATH='/sracth/dart/models/teacher_ft_lr1e-5_epc16_bsz8/best_tfmr/model_3k_raw_out_6_samp.test'

export REF_PATH
echo $MODEL_NAME
echo $REF_PATH
rm $SLURM_TMPDIR/model.out
python3 dart/run_calc_ppl.py \
  --model_name $MODEL_NAME\
  --input_path /sracth/seq2seq_nlg/dart_eval_data/test.src \
  --reference_path $REF_PATH