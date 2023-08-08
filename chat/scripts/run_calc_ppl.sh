#!/bin/sh

rm $SLURM_TMPDIR/model.out
CUDA_LAUNCH_BLOCKING=1 python3 /chat/run_calc_ppl.py \
    --reference_path scratch/chat/models/teacher_ft_lr1e-4_medium/best_tfmr/model_out_b1_T1._ML3.test\
    --model_name scratch/chat/models_freeze/student_4L_rkl_sbz1_p0.9_lr1e-4_dm/best_tfmr/\
    --input_path scratch/chat/Commonsense-Dialogues/test.source\
    --num_beams 1