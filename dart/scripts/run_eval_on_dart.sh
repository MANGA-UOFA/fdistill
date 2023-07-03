#! /bin/bash

OUTPUT_PATH=""
OUTPUT_FILE=${OUTPUT_PATH}model_out_6.test
echo $OUTPUT_FILE

TEST_TARGETS_REF0=dart_eval_data/test.ref0
TEST_TARGETS_REF1=dart_eval_data/test.ref1
TEST_TARGETS_REF2=dart_eval_data/test.ref2

# BLEU
./multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} < ${OUTPUT_FILE} > bleu.txt

python prepare_files.py ${OUTPUT_FILE} ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2}
# METEOR
cd meteor-1.5/ 
java -Xmx2G -jar meteor-1.5.jar ${OUTPUT_FILE} ../all-notdelex-refs-meteor.txt -l en -norm -r 8 > ../meteor.txt
cd ..

# TER
cd tercom-0.7.25/
java -jar tercom.7.25.jar -h ../relexicalised_predictions-ter.txt -r ../all-notdelex-refs-ter.txt > ../ter.txt
cd ..

# MoverScore
python moverscore.py ${TEST_TARGETS_REF0} ${OUTPUT_FILE} > moverscore.txt
# BERTScore

OUTPUT_FILE=${OUTPUT_PATH}model_raw_out_6.test

TEST_TARGETS_REF0=/home/lcc/scratch/seqkd/seq2seq_nlg/dart_eval_data_untknzed/test.ref0
TEST_TARGETS_REF1=/home/lcc/scratch/seqkd/seq2seq_nlg/dart_eval_data_untknzed/test.ref1
TEST_TARGETS_REF2=/home/lcc/scratch/seqkd/seq2seq_nlg/dart_eval_data_untknzed/test.ref2

python /home/nlg/evaluation/bert_score/bert_score_cli/score.py -l 8 --model "nlg/models/bert-base-uncased/" -r ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} -c ${OUTPUT_FILE} --lang en > bertscore.txt
# BLEURT
#python -m bleurt.score -candidate_file=${OUTPUT_FILE} -reference_file=${TEST_TARGETS_REF0} -bleurt_checkpoint=bleurt/bleurt/test_checkpoint -scores_file=bleurt.txt

python print_scores.py

python -m bleurt.score_files   \
        -candidate_file=${OUTPUT_FILE}\
        -reference_file=${TEST_TARGETS_REF0} \
        -bleurt_checkpoint=./bleurt/bleurt/test_checkpoint/ > tmp.score

python -c "import numpy as np; import pandas as pd; print('bleurt score: ', np.mean(pd.read_csv('./tmp.score', header=None)) )"