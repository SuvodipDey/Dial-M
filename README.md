# Dial-M: A Masking-based Framework for Dialogue Evaluation
Official code repo of Dial-M metric for dialogue evaluation.

## Install dependencies
python 3.10
```console
❱❱❱ pip install -r requirements.txt
```

## Download Datasets
Download the following datasets. 

1. DailyDialog: Download link http://yanran.li/files/ijcnlp_dailydialog.zip
2. PersonaChat: Download data using ParlAI (https://parl.ai/docs/tasks.html#persona-chat)
3. Wizard-of-Wikipedia: Download data using ParlAI (https://parl.ai/docs/tasks.html#wizard-of-wikipedia)
4. Topical-Chat: Download data by following the instructions of https://github.com/alexa/Topical-Chat. Also, download TopicalChatEnriched following https://github.com/alexa/Topical-Chat/tree/master/TopicalChatEnriched.
5. MultiWOZ 2.1: Download data from https://github.com/budzianowski/multiwoz/tree/master/data followed by the required pre-processing instructions given in https://github.com/budzianowski/multiwoz.

Set the dataset paths correctly in the following files: prepare_dataset_mlm.py and prepare_dataset_dialm.py

## Pre-training 
Finetune RoBERTa-base model on MLM task with different dialogue datasets. 
```console
❱❱❱ python train_mlm.py -path=<output_mlm> -epochs=30
```
Model checkpoint: https://drive.google.com/file/d/1q9vim2_goV-sXW05sF7bgWOLuE61mC6h/view?usp=drive_link

## Finetuning 
### Dial-M (Main model) 
```console
❱❱❱ python train_dialm.py -pre=<output_mlm> -path=<output_dialm> -epochs=10
```
Model checkpoint: https://drive.google.com/file/d/1lkuXjgxBfEbizs8jRVLLJg58k_msmjqy/view?usp=drive_link

### Models for Ablation Study
1. Dial-M with random token masking instead of keyword masking.
```console
❱❱❱ python train_dialm.py -pre=<output_mlm> -path=<output_dialm_random> -epochs=10 -random
```

2. Dial-M without pre-training on dialogue datasets. Use roberta-base as the pre-trained model.
```console
❱❱❱ python train_dialm.py -pre=roberta-base -path=<output_dialm_nopre> -epochs=10 -no_pre
```

## Evaluation
Evaluations are performed on USR (https://doi.org/10.18653/v1/2020.acl-main.64), PredictiveEngage (https://ojs.aaai.org/index.php/AAAI/article/view/6283), and HolisticEval (https://aclanthology.org/2020.acl-main.333/) datasets. The evaluation data is available in the "evaluation_data.zip" file. Unzip the file before running the evaluation scripts.

### Main Result
Dial-M (with pre-training and finetuning)
```console
❱❱❱ python eval_dialm.py -path=<output_dialm> -out=<out_dir>
```

Evaluation of different sub-metric on USR dataset.
```console
❱❱❱ python eval_sub_metric.py -path=<output_dialm> -out=<out_dir>
```

### Ablation Study
1. Dial-M with random token masking (instead of keyword masking) during finetuning.
```console
❱❱❱ python eval_dialm.py -path=<output_dialm_random> -out=<out_dir>
```

2. Dial-M without pre-training on dialogue datasets.
```console
❱❱❱ python eval_dialm.py -path=<output_dialm_nopre> -out=<out_dir> -no_pre
```

3. Dial-M with pre-training but no finetuning.
```console
❱❱❱ python eval_dialm.py -path=<output_mlm> -out=<out_dir>
```

4. Dial-M without pre-training and finetuning i.e. the score is computed with roberta-base model.
```console
❱❱❱ python eval_dialm.py -path=roberta-base -out=out
```
