# Pipelines for Training and Evaluation


## Data-to-text (DART)

* Folder: `/dart/` 
* Datasets: `dart` ([Google drive link](https://drive.google.com/file/d/1V7bPndyoTQxcJ6m1BoXAw7-ub-jv8Wh1/view?usp=sharing))
* Teacher model
  * Init: `bart-large`
  * Fine-tune: `scripts/run_finetune_teacher.sh` ([Google drive link](https://drive.google.com/file/d/12C5vkYoxMXdRktcLlYLcT0fUHKj6Udwd/view?usp=sharing))
* Student init model:
  * Pre-train init student model: `scripts/run_pretrain_student_distill.sh` ([Google drive link]())
* Generate pseudo-target with Teacher model (useful for Seqkd, JS, TVD) 
  * `scripts/run_teacher_label_all.sh` ([Google drive link](https://drive.google.com/file/d/14NSX8diOZEFIFsZRB5EGtq5l1wKc7ygt/view?usp=sharing)), replace `train.json`
* Run KD methods (`/scripts/`)
  * Seqkd: `run_seqkd.sh`
  * ENGINE: `run_engine.sh`
  * RKL: `run_rkl.sh`
  * KL: `run_kl_sample.sh`
  * JS: `run_js.sh`
  * TVD: `run_tvd_symm.sh`
* Decode (`/scripts/`) ([Google drive link](https://drive.google.com/file/d/1V7bPndyoTQxcJ6m1BoXAw7-ub-jv8Wh1/view?usp=sharing))
  * `run_eval.sh`
* Eval
  * `cd /evaluation/` ([Google drive link](https://drive.google.com/file/d/1at5kY8YT-7yxOZdI7aRn-JCTBm0tdAzT/view?usp=sharing))
  * `sh ./run_eval_on_dart.sh` (need to modify the `$OUTPUT_FILE` and download [bert-base-uncased model](https://huggingface.co/bert-base-uncased/tree/main))

* Calculate coverage loss (PPL of teacher)
  * run `python3 dart/run_calc_ppl.py --reference_path [teacher_output_path] --input_path [input_path] --model_name [student_model_path] --save_path /tmp/`
* Calculate likelihood loss (PPL of student)
  * run `python3 dart/run_calc_ppl.py --model_name [teacher_model_path] --input_path [input_path] --reference_path [student_output_path] --save_path /tmp/`



## Summarization

* Folder: `/summa/`
* Dataset: `xsum` (`wget https://cdn-datasets.huggingface.co/summarization/xsum.tar.gz`)
* Teacher model: `https://huggingface.co/facebook/bart-large-xsum`
* Student init model:
  * Pre-train init student model: `run_pretrain_student_distill.sh` ([Google drive link](https://drive.google.com/file/d/1V9dCXNEvv1ttzUv4pCzOeF1JjR96cDei/view?usp=sharing))
* Generate pseudo-target with Teacher model (useful for Seqkd, JS, TVD) 
  * `run_teacher_label.sh` ([Google drive link](https://drive.google.com/file/d/132ryNhjWsR3FquBEe2QLBW_Bx_RR5jky/view?usp=sharing)), replace `train.target`
* Run KD methods (`/scripts/`)
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

## Machine Translation (WMT16 EN-RO, 100k training data)

* Folder: `/t5mt/`
* Dataset: `wmt_en_ro_100k` ([Google drive link](https://drive.google.com/file/d/1s0ONfQubaXdjvGtg3ijsmD1W8-k93WgD/view?usp=sharing))
* Teacher model
  * Init: `t5-base`
  * Fine-tune: `scripts_sm/run_finetune_teacher.sh`
* Student init model:
  * Pre-train init student model: `scripts_sm/run_pretrain_student_distill.sh` ([Google drive link](https://drive.google.com/file/d/17h4nsC0heXBV-Ug52aN1Q4TefoUGRCJz/view?usp=sharing))
* Generate pseudo-target with Teacher model (useful for Seqkd, JS, TVD) 
  * `scripts_sm/run_teacher_label.sh` ([Google drive link](https://drive.google.com/file/d/18kK2Ju097jdNc_L5lSrbEc2j7kFVuQCu/view?usp=sharing)), replace `train.target`
* Run KD methods (`/scripts/`)
  * Seqkd: `run_seqkd.sh`
  * ENGINE: `run_engine.sh`
  * RKL: `run_rkl.sh`
  * KL: `run_kl.sh`
  * JS: `run_js.sh`
  * TVD: `run_tvd_symm.sh`
* Decode and eval
  * `sh scripts_sm/run_eval.sh`
* Calculate coverage loss (PPL of teacher)
  * run `python3 t5mt/run_calc_ppl.py --reference_path [teacher_output_path] --input_path [input_path] --model_name [student_model_path] --save_path /tmp/`
* Calculate likelihood loss (PPL of student)
  * run `python3 t5mt/run_calc_ppl.py --model_name [teacher_model_path] --input_path [input_path] --reference_path [student_output_path] --save_path /tmp/`

## Chat

* Folder: `/chat/`
* Dataset: `Commonsense-Dialogues` ([Google drive link](https://drive.google.com/file/d/1AhND-wEgyidEaOeAn6WIFzZ1znnLvmBS/view?usp=sharing))
* Teacher model
  * Init: `microsoft/DialoGPT-medium` 
  * Fine-tune: `scripts/run_finetune_teacher.sh` ([Google drive link](https://drive.google.com/file/d/18h-8lqYy-Wb-i47Aa6w3KJXtecCwekRr/view?usp=sharing))
* Student init model:
  * Pre-train init student model: `scripts/run_pretrain_student_distill.sh` ([Google drive link](https://drive.google.com/file/d/1fiGY-_fYGkAOJ4FW-gQn4Z1vTJCUwwOp/view?usp=sharing))
* Generate pseudo-target with Teacher model (useful for Seqkd, JS, TVD) ([Google drive link](https://drive.google.com/file/d/1JkyGR_xdzW7F0220wXQgY6hZG6CE7Sjj/view?usp=sharing)), replace `train.target`
  * `scripts/run_teacher_label.sh`
* Run KD methods (`/scripts/`)
  * Seqkd: `run_seqkd.sh`
  * ENGINE: `run_engine.sh`
  * RKL: `run_rkl.sh`
  * KL: `run_kl.sh`
  * JS: `run_js.sh`
  * TVD: `run_tvd_symm.sh`
* Decode and eval
  * sh `scripts/run_eval.sh` (need to download [bert-base-uncased model](https://huggingface.co/bert-base-uncased/tree/main))
* Calculate coverage loss (PPL of teacher)
  * run `python3 chat/run_calc_ppl.py --reference_path [teacher_output_path] --input_path [input_path] --model_name [student_model_path] --save_path /tmp/`
* Calculate likelihood loss (PPL of student)
  * run `python3 chat/run_calc_ppl.py --model_name [teacher_model_path] --input_path [input_path] --reference_path [student_output_path] --save_path /tmp/`

# Acknowledgements
* The methods in our codebase are mainly implemented with PyTorch and [Huggingface's Transformers libararies](https://github.com/huggingface/transformers/tree/v4.1.0)
* The pre-distillation part is based on the method proposed in Shleifer & Rush (2020) and [their implementation](https://github.com/huggingface/transformers/tree/v4.1.0/examples/research_projects/seq2seq-distillation)
* We use [Maluuba's nlg-eval](https://github.com/Maluuba/nlg-eval) to measure the BLEU score for the dialogue task.
