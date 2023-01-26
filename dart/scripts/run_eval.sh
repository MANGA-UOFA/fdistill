python3 ../run_eval.py \
  --model_name $MODEL_NAME\
  --input_path '' \
  --save_path '' \
  --reference_path '' \
  --score_path '' \
  --device cuda --num_beams $BEAM
  "$@"