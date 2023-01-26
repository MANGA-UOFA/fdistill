#!/bin/sh


export TO_DIR='/'

export ORG_DIR='/'
cp $ORG_DIR/ $TO_DIR/ -r

python3 /home/fdill/t5mt/run_eval.py \
  --model_name /home/fdill/t5mt/models/teacher_en-ro_epc9/best_tfmr/ \
  --input_path $ORG_DIR/train.source\
  --save_path $TO_DIR/train.target \
  --reference_path $ORG_DIR/train.target \
  --score_path $SLURM_TMPDIR/_tmp.json \
  --max_input_length 300\
  --device cuda --num_beams 5\
  "$@"

#cp $SLURM_TMPDIR/teacher.out $TO_DIR/train.target
