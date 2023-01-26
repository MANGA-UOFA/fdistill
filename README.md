# f-Divergence Minimization for Sequential Knowledge Distillation


## Data-to-text (DART)

* Folder: `/dart/` 
* Datasets: `dart`
* Teacher model
  * Init: `bart-large`
  * Fine-tune: `scripts/run_finetune_teacher.sh`
* Student init model:
  * MLE + WordKL + Hidden-distill
  * Pre-train init student model: `scripts/run_pretrain_student_distill.sh`
* Generate pseudo-target with Teacher model (useful for Seqkd, KL, JS, TVD)
  * `scripts/run_teacher_label_all.sh`
* Run KD methods (`dart/scripts/`)
  * Seqkd: `run_seqkd.sh`
  * ENGINE: `run_engine.sh`
  * RKL: `run_rkl.sh`
  * KL: `run_kl_sample.sh`
  * JS: `run_js.sh`
  * TVD: `run_tvd_symm.sh`
* Decode (`/scripts/`)
  * `run_eval.sh`
* Eval
  * We follow the original DART paper for evaluation
    * `cd /evaluation/` ([link](https://github.com/Yale-LILY/dart/tree/master/evaluation))
    * `sh ./run_eval_on_dart.sh` (need to modify the `$OUTPUT_FILE`)
* Calculate coverage loss (PPL of teacher)
  * run `python3 dart/run_calc_ppl.py --reference_path [teacher_output_path] --input_path [input_path] --model_name [student_model_path] --save_path /tmp/`
* Calculate likelihood loss (PPL of student)
  * run `python3 dart/run_calc_ppl.py --model_name [teacher_model_path] --input_path [input_path] --reference_path [student_output_path] --save_path /tmp/`

## Summarization (XSum)

* Folder: `/summa/`
* Dataset: `xsum`
* Teacher model: `https://huggingface.co/facebook/bart-large-xsum`
* Student init model:
  * MLE + WordKL + Hidden-distill
  * pre-train init student model: `run_pretrain_student_distill.sh`
* Generate pseudo-target with Teacher model (useful for Seqkd, KL, JS, TVD)
  * `run_teacher_label.sh`
* Run KD methods (`summa/scripts/`)
  * Seqkd: `run_seqkd.sh`
  * ENGINE: `run_engine.sh`
  * RKL: `run_rkl.sh`
  * KL: `run_kl.sh`
  * JS: `run_js.sh`
  * TVD: `run_tvd_symm.sh`
* Decode and Evaluate
  * run `eval.sh`
* Calculate coverage loss (PPL of teacher)
  * run `python3 summa/run_calc_ppl.py --reference_path [teacher_output_path] --input_path [input_path] --model_name [student_model_path] --save_path /tmp/`
* Calculate likelihood loss (PPL of student)
  * run `python3 summa/run_calc_ppl.py --model_name [teacher_model_path] --input_path [input_path] --reference_path [student_output_path] --save_path /tmp/`
  
## Machine Translation (WMT16 EN-RO)

* Folder: `/t5mt/` 
* Dataset: `wmt_en_ro_100k`
* Teacher model
  * Init: `mbart-large-cc25` ([link](https://huggingface.co/facebook/mbart-large-cc25/tree/main))
  * Fine-tune: `scripts_sm/run_finetune_teacher.sh`
* Student init model:
  * Pre-train init student model: `scripts_sm/run_distill_pkd.sh` (or `scripts_sm/run_pretrain_student_distill.sh`, different ways of calculating hidden loss).
* Generate pseudo-target with Teacher model (useful for Seqkd, KL, JS, TVD)
  * `scripts_sm/run_teacher_label.sh`
* Run KD methods (`/scripts_sm/`)
  * Seqkd: `run_seqkd.sh`
  * ENGINE: `run_engine.sh`
  * RKL: `run_rkl.sh`
  * KL: `run_kl.sh`
  * JS: `run_js.sh`
  * TVD: `run_tvd_symm.sh`
* Decode and eval
  * `sh scripts_sm/run_eval.sh`

## Dialogue Generation (Commensense Dialogue)

* Folder: `/chat/`
* Dataset: `Commonsense-Dialogues`
* Teacher model
  * Init: `microsoft/DialoGPT-medium`
  * Fine-tune: `scripts/run_finetune_teacher.sh`
* Student init model:
  * Pre-train init student model: `scripts/run_pretrain_student_distill.sh`
* Generate pseudo-target with Teacher model (useful for Seqkd, KL, JS, TVD)
  * `scripts/run_teacher_label.sh`
* Run KD methods (`/scripts/`)
  * Seqkd: `run_seqkd.sh`
  * ENGINE: `run_engine.sh`
  * RKL: `run_rkl.sh`
  * KL: `run_kl.sh`
  * JS: `run_js.sh`
  * TVD: `run_tvd_symm.sh`

* Decode and eval
  * sh `scripts/run_eval.sh`
