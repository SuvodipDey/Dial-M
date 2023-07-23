import os
import pandas
import math
import random
import torch
import time
from transformers import AutoTokenizer, RobertaForMaskedLM
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler
import argparse
import re
import collections
from scipy.stats import pearsonr, spearmanr
from prepare_eval_data import load_tc_usr, load_pc_usr, load_engage_data, load_holistic_data
import torch.nn.functional as F
import spacy
from nltk.corpus import stopwords
import shutil
import yake

#----------------------------

"""
Main Result:-
Dial-M (with pre-training and finetuning)
python eval_dialm.py -path=<output_dialm> -out=<out_dir>

Ablation Study:-
1. Dial-M with random token masking (instead of keyword masking) during finetuning
python eval_dialm.py -path=<output_dialm_random> -out=<out_dir>

2. Dial-M without pre-training on dialogue datasets
python eval_dialm.py -path=<output_dialm_nopre> -out=<out_dir> -no_pre

3. Dial-M with pre-training but no finetuning
python eval_dialm.py -path=<output_mlm> -out=<out_dir>

4. Dial-M without pre-training and finetuning i.e. the score is computed with roberta-base model
python eval_dialm.py -path=roberta-base -out=out

"""

#----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-path','--path', help='path of the model directiory', required=True)
parser.add_argument('-out','--out', help='name of output directory', required=True)
parser.add_argument('-lbl','--lbl', help='file label', required=False, default="")
parser.add_argument('-keys','--keys', help='Number of keywords', type=int, required=False, default=20)
parser.add_argument('-no_pos','--no_pos', help='Skip POS keywords', default=True, action='store_false')
parser.add_argument('-no_pre','--no_pre', help='Inference for model with no pre-training', default=False, action='store_true')
args = vars(parser.parse_args())
model_dir = args['path']
numOfKeywords = args['keys']
add_pos = args['no_pos']
out_label = args['lbl']
no_pre = args['no_pre']

if(model_dir=="roberta-base"):
    print("Evaluating with roberta-base model.")
    model_dir = "roberta-base"
    out_dir = "result_roberta_base"
    no_pre = True
else:
    if(not os.path.isdir(model_dir)):
        print("Model Directory does not exist.")
        exit(0)
    else:
        out_dir = os.path.join(model_dir, args['out'])
    
if(not os.path.isdir(out_dir)):
    os.mkdir(out_dir)

print(f"Model directory: {model_dir}")
print(f"Output directory: {out_dir}")
print(f"numOfKeywords: {numOfKeywords}")
print(f"no_pre: {no_pre}")
    
#----------------------------

eou_token = "<eou>"
knlg_token = "<knlg>"
max_len=512

lst_pos_tags = ['NN', 'NNP', 'NNS', 'JJ', 'CD', 'VB', 'VBN', 'VBD', 'VBG', 'RB', 'VBP', 'VBZ', 'NNPS', 'JJS']
stop_words = stopwords.words('english')

SEED = 10
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)      
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
if(no_pre):
    eou_token = tokenizer.eos_token
model = RobertaForMaskedLM.from_pretrained(model_dir)
model.to(device)
model.eval()

language = "en"
max_ngram_size = 1
deduplication_threshold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
nlp = spacy.load("en_core_web_sm")
print("Model Loaded")

#----------------------------

def tokenize_sentence(txt, tokenizer):
    result = tokenizer(txt)
    word_ids = result.word_ids()
    if tokenizer.is_fast:
        result["word_ids"] = [word_ids[i] for i in range(len(result["input_ids"]))]
        for i in range(len(result["input_ids"])):
            if(result["input_ids"][i] >= 50265):
                result["word_ids"][i] = None
                break
    return result

def tokenize_sentence_truncated(txt, tokenizer, n):
    result = tokenizer(txt, truncation=True, max_length=n)
    word_ids = result.word_ids()
    if tokenizer.is_fast:
        result["word_ids"] = [word_ids[i] for i in range(len(result["input_ids"]))]
        for i in range(len(result["input_ids"])):
            if(result["input_ids"][i] >= 50265):
                result["word_ids"][i] = None
                break
    return result

def get_word_mapping(tok):
    word_ids = tok["word_ids"].copy()
    mapping = collections.defaultdict(list)
    current_word_index = -1
    current_word = None
    for idx, word_id in enumerate(word_ids):
        if word_id is not None:
            if word_id != current_word:
                current_word = word_id
                current_word_index += 1
            mapping[current_word_index].append(idx)
    return mapping

def get_pos_tags(doc, m_type):
    pos_tags = {}
    for token in doc:
        if(m_type==1):
            if(not (token.is_stop or token.is_punct or token.is_space or token.text.lower() in stop_words)):
                if(token.tag_ in lst_pos_tags):
                    pos_tags[token.text.lower()] = token.tag_
        else:
            if(not (token.is_punct or token.is_space)):
                pos_tags[token.text.lower()] = token.tag_
    return pos_tags

def get_mask_words(txt, tok, mapping, add_pos):
    yake_doc = txt.replace(eou_token, " ")
    yake_doc = yake_doc.strip()
    keywords = custom_kw_extractor.extract_keywords(yake_doc)
    lst_kw = [kw[0] for kw in keywords]
    
    txt_doc = nlp(txt)
    if(len(lst_kw)<numOfKeywords and add_pos):
        n = numOfKeywords-len(lst_kw)
        pos_tags = get_pos_tags(txt_doc, 1)
        for w in pos_tags:
            if(w not in lst_kw):
                lst_kw.append(w)
                n = n-1
                if(n==0):
                    break
    
    mask = []
    mask_words = []
    for idx in mapping:
        start, end = tok.word_to_chars(idx)
        word = txt[start:end].lower()
        if word in lst_kw:
            mask.append(idx)
            mask_words.append(word)
            
    if(len(mask)==0):
        lst_kw = []
        n = numOfKeywords
        pos_tags = get_pos_tags(txt_doc, 2)
        for w in pos_tags:
            lst_kw.append(w)
            n = n-1
            if(n==0):
                break
                
        for idx in mapping:
            start, end = tok.word_to_chars(idx)
            word = txt[start:end].lower()
            if word in lst_kw:
                mask.append(idx)
                mask_words.append(word)
                
        if(len(mask)==0):
            for idx in mapping:
                start, end = tok.word_to_chars(idx)
                word = txt[start:end].lower()
                mask.append(idx)
                mask_words.append(word)
            
    return mask, mask_words
    
def get_masked_tokens(tokenizer, tok, mapping, mask):
    mask_ids = []
    input_ids = tok["input_ids"].copy()
    labels = [-100]*len(input_ids)
    for word_id in mask:
        for idx in mapping[word_id]:
            mask_ids.append(input_ids[idx])
            labels[idx] = input_ids[idx]
            input_ids[idx] = tokenizer.mask_token_id
    return input_ids, labels

def evaluate(input_id, lbl, attn_mask):
    input_ids = torch.tensor([input_id], dtype=torch.long).to(device)
    labels = torch.tensor([lbl], dtype=torch.long).to(device)
    attention_masks = torch.tensor([attn_mask], dtype=torch.long).to(device)
    loss = 0.0
    with torch.no_grad():
        output = model(input_ids = input_ids, attention_mask = attention_masks, labels = labels)
        loss = output.loss.item()
    return loss

def get_score(prev, resp, tok_context, tok_prev, tok_resp, tok_condition, use_condition, logger):
    map_resp = get_word_mapping(tok_resp)
    mask, mask_words = get_mask_words(resp, tok_resp, map_resp, add_pos)
    
    score = -1
    if(len(mask)>0):
        total_score = 0
        for word_id in mask:
            resp_masked, lbl_resp = get_masked_tokens(tokenizer, tok_resp, map_resp, [word_id])
            tok1 = []
            j=0
            if(len(tok_context)>0):
                tok1.extend(tok_context.copy()[j:-1])
                j=1
            if(prev is not None):
                tok1.extend(tok_prev["input_ids"].copy()[j:-1])
                j=1
            lbl1 = [-100]*len(tok1)
            tok1.extend(resp_masked[j:])
            lbl1.extend(lbl_resp[j:])

            if(use_condition):
                tok_kn = tok_condition["input_ids"].copy()
                tok_kn[0] = tokenizer.sep_token_id
                tok1.extend(tok_kn)
                lbl1.extend([-100]*len(tok_kn))
            attn1 = [1]*len(tok1)
            score = evaluate(tok1, lbl1, attn1)
            total_score+= score
        score = total_score/len(mask)
    return round(score, 4)

def get_metric(txt_input, response, logger):
    utt_list = []
    context = None
    condition = None
    use_condition = False
    
    if(knlg_token in txt_input):
        arr  = txt_input.split(knlg_token)
        context  = arr[0].strip()
        condition = arr[1].strip()
        use_condition = True
    else:
        context = txt_input
        
    arr = context.split(eou_token)
    utt_list = []
    for i in range(len(arr)-1):
        utt = arr[i].strip()
        utt_list.append(utt)
    
    if(response is not None):
        resp = response
    else:
        resp = utt_list[-1]
    prev = None
    if(len(utt_list)>1):
        prev = utt_list[-2]
    
    resp = f"{resp}{eou_token}"
    tok_resp = tokenize_sentence(resp, tokenizer)
    
    tok_prev = []
    if(prev is not None):
        prev = f"{prev}{eou_token}"
        tok_prev = tokenize_sentence(prev, tokenizer)
    
    tok_condition = None
    tok_count = 0
    if(use_condition):
        tok_condition = tokenize_sentence(condition, tokenizer)
        if(prev is not None):
            tok_count = len(tok_prev["input_ids"]) + len(tok_resp["input_ids"]) + len(tok_condition["input_ids"]) - 2
        else:
            tok_count = len(tok_resp["input_ids"]) + len(tok_condition["input_ids"]) - 2
    else:
        if(prev is not None):
            tok_count = len(tok_prev["input_ids"]) + len(tok_resp["input_ids"]) - 2
        else:
            tok_count = len(tok_resp["input_ids"]) - 2
        
    if(tok_count>max_len):
        if(use_condition):
            if(prev is not None):
                n = max_len - len(tok_resp["input_ids"]) - len(tok_resp["input_ids"])
            else:
                n = max_len - len(tok_resp["input_ids"])
            tok_condition = tokenize_sentence_truncated(condition, tokenizer, n)
        else:
            print(f"Input length exceeded!!! {tok_count}")
            logger.write("Input length exceeded!!!\n")
            logger.write(f"tok_count: {tok_count}\n")
            h = len(tok_resp["input_ids"])
            logger.write(f"tok_resp: {h}\n")
            h = len(tok_prev["input_ids"])
            logger.write(f"tok_prev: {h}\n")
            return -100
    
    tok_context = []
    context = ""
    if(len(utt_list)>2):
        con_list = []
        n = 0
        for k in range(len(utt_list)-3,-1,-1):
            utt_text = f"{utt_list[k]}{eou_token}"
            tok_utt = tokenizer(utt_text)
            if(n+len(tok_utt["input_ids"])+tok_count-2<=max_len):
                n += len(tok_utt["input_ids"])-2
                con_list.append(utt_text)
            else:
                break
        con_list.reverse()            
        context = "".join(con_list)
        tok_context = tokenizer(context)["input_ids"]
    
    logger.write(f"prev: {prev}\n")
    logger.write(f"resp: {resp}\n")
    
    score = get_score(prev, resp, tok_context, tok_prev, tok_resp, tok_condition, use_condition, logger)
    logger.write(f"Dial-M score: {score}\n")
    return score

def compute_correlation(lst_gt, lst_score, logger):
    corr1, p_val1 = pearsonr(lst_gt, lst_score)
    corr2, p_val2 = spearmanr(lst_gt, lst_score)
    logger.write(f"Correlation between GT and Dial-M score: Pearson = ({round(corr1,4)}, {round(p_val1,4)}), Spearman = ({round(corr2,4)}, {round(p_val2,4)})\n")
    
    logger.write("-"*30+"\n")
    logger.write("-"*30+"\n")
        
#----------------------------

def evaluate_usr(dataset):
    out_path = os.path.join(out_dir, f'out_{dataset}{out_label}.txt')
    logger = open(out_path, "w")
    
    if("topical" in dataset):
        usr_eval = load_tc_usr(eou_token, knlg_token)
    elif("persona" in dataset):
        usr_eval = load_pc_usr(eou_token, knlg_token)
    else:
        logger.write("Unknown dataset ...\n")
        return
        
    f_gt = []
    f_score = []
    
    no_key = 0
    t_count = 0
    conv = 0
    oob = 0
    for i in range(len(usr_eval)):
        t_count+=1
        responses = usr_eval[i]['responses']
        context = usr_eval[i]['ctx']
        lst_gt = []
        lst_score = []
        
        for j in range(len(responses)):
            if responses[j]['model'] == 'Original Ground Truth':
                continue
            response = responses[j]['response']
            response = response.replace("\n","")
            overall_score = responses[j]['Overall']
            v = sum(overall_score)/len(overall_score)
            logger.write(f"Overall score: Avg. {v} : {overall_score}\n")
            
            score = get_metric(context, response, logger)
            if(score==-100):
                oob+=1
                logger.write(f"Token out of bound !!! {i}-{j}\n")
                logger.write("-"*30+"\n")
            else:
                logger.write("-"*30+"\n")
                if(score>-1):
                    lst_gt.append(v)
                    f_gt.append(v)
                    lst_score.append(score)
                    f_score.append(score)
                else:
                    no_key +=1
        
        conv+=1
        compute_correlation(lst_gt, lst_score, logger)
        #if(conv==2):
        #    break

    logger.write("="*50+"\n")
    compute_correlation(f_gt, f_score, logger)
    logger.write(f"total: {t_count}\n")
    logger.write(f"out of bound: {oob}\n")
    logger.write(f"no keywords: {no_key}\n")
    print(f"Evaluation of USR {dataset} done.")

#----------------------------

def evaluate_grade_fed(dataset, data_eval):
    out_path = os.path.join(out_dir, f'out_{dataset}{out_label}.txt')
    logger = open(out_path, "w")
        
    f_gt = []
    f_score = []
    
    no_key = 0
    t_count = 0
    conv = 0
    for i in range(len(data_eval)):
        t_count+=1
        context = data_eval[i]['ctx']
        overall_score = data_eval[i]['Overall']
        if(isinstance(overall_score, list)):
            v = sum(overall_score)/len(overall_score)
            logger.write(f"Overall score: Avg. {v} : {overall_score}\n")
        else:
            v = overall_score
            logger.write(f"Overall score: Avg. {v}\n")
        
        score = get_metric(context, None, logger)
        logger.write("-"*30+"\n")
        
        if(score>-1):
            f_gt.append(v)
            f_score.append(score)
        else:
            no_key +=1

    logger.write("="*50+"\n")
    compute_correlation(f_gt, f_score, logger)
    logger.write(f"total: {t_count}\n")
    logger.write(f"no keywords: {no_key}\n")
    print(f"Evaluation of {dataset} done.")
    
#----------------------------

evaluate_usr("topical")
evaluate_usr("persona")

engage_eval = load_engage_data(eou_token)
evaluate_grade_fed("pred_engage", engage_eval)

holistic_eval = load_holistic_data(eou_token)
evaluate_grade_fed("holistic_eval", holistic_eval)

print("done")

#----------------------------