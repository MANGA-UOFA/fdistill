#!/bin/sh

export MODEL_NAME='/home/fdill/chat/models/student_4L_mle/best_tfmr/'
export BEAM=1
export TEMP=1.
export MIN_LEN=3
echo $MODEL_NAME
rm $SLURM_TMPDIR/model.out
CUDA_LAUNCH_BLOCKING=1 python3 /home/fdill/chat/run_eval.py \
  --model_name $MODEL_NAME\
  --input_path /home/fdill/chat/Commonsense-Dialogues/test.source\
  --save_path $SLURM_TMPDIR/model.out \
  --reference_path /home/fdill/chat/Commonsense-Dialogues/test.target\
  --score_path $SLURM_TMPDIR/metrics.json \
  --device cuda --num_beams $BEAM --temperature $TEMP --min_length $MIN_LEN --do_trunc_pred\
  "$@"

cp $SLURM_TMPDIR/model.out $MODEL_NAME"model_out_b${BEAM}_T${TEMP}_ML${MIN_LEN}.test"
cp $SLURM_TMPDIR/metrics.json $MODEL_NAME"test_metrics_t${TEMP}_ML${MIN_LEN}.json"

python3 /home/fdill/nlg/evaluation/bert_score/bert_score_cli/score.py \
        -l 8 --model "/home/fdill/nlg/models/bert-base-uncased/" \
        -r /home/fdill/chat/Commonsense-Dialogues/test.target  \
        -c $SLURM_TMPDIR/model.out \
        --lang en

nlg-eval --hypothesis=$SLURM_TMPDIR/model.out \
         --references=/home/fdill/chat/Commonsense-Dialogues/test.target