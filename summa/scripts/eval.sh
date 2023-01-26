export MODEL_NAME=''
echo $MODEL_NAME
export BEAM=6
python3 ../run_eval.py \
  --model_name $MODEL_NAME \
  --input_path '' \
  --save_path '' \
  --reference_path '' \
  --score_path '' \
  --num_beams $BEAM\
  --length_penalty 0.5\
  --device cuda
  "$@"
