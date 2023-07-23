import os
import math
import random
import torch
import time
from transformers import AutoTokenizer, RobertaForMaskedLM, AdamW
from torch.utils.data import TensorDataset
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler
import logging
import argparse
import re
import collections
from tqdm.auto import tqdm
from transformers import get_scheduler
from prepare_dataset_reval import get_samples
from transformers import pipeline
import pandas as pd
import spacy
from nltk.corpus import stopwords
import string
import shutil
import yake

#----------------------------
"""
Dial-M (Main Model):-
python train_dialm.py -pre=<output_mlm> -path=<output_dialm> -epochs=10

Ablation Study:-
1. Dial-M with random token masking instead of keyword masking.
python train_dialm.py -pre=<output_mlm> -path=<output_dialm_random> -epochs=10 -random

2. Dial-M without pre-training on dialogue datasets. Use roberta-base as the pre-trained model.
python train_dialm.py -pre=roberta-base -path=<output_dialm_nopre> -epochs=10 -nopre

"""
#----------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-pre','--pre', help='path of the pre-trained model directiory', required=True)
parser.add_argument('-path','--path', help='path of the model/output directiory', required=True)
parser.add_argument('-src_file','--src_file', help='path of the source file', required=True)
parser.add_argument('-pos','--pos', help='Add POS keywords', default=False, action='store_true')
parser.add_argument('-random','--random', help='Mask random words', default=False, action='store_true')
parser.add_argument('-keys','--keys', help='Number of keywords', type=int, required=False, default=20)
parser.add_argument('-max_len','--max_len', help='Maximum input length', type=int, required=False, default=256)
parser.add_argument('-epochs','--epochs', help='Number of training epochs', type=int, required=False, default=10)
parser.add_argument('-batch','--batch', help='Batch size', type=int, required=False, default=96)
parser.add_argument('-lr','--lr', help='Learning rate', type=float, required=False, default=1e-5)
parser.add_argument('-no_pre','--no_pre', help='Inference for model with no pre-training', default=False, action='store_true')

args = vars(parser.parse_args())
model_pretrained = args['pre']
model_dir = args['path']
src_file = args['src_file']
add_pos = args['pos']
mask_random = args['random']
max_keyword = args['keys']
max_len = args['max_len']
num_train_epochs = args['epochs']
BATCH_SIZE = args['batch']
learning_rate = args['lr']
no_pre = args['no_pre']

if(os.path.isdir(model_dir)):
    print("Model Directory exists.")
    exit(0)
else:
    os.mkdir(model_dir)
    print(f"Model directory {model_dir} created.")
shutil.copy(src_file, model_dir)

#----------------------------

eou_token = "<eou>"
knlg_token = "<knlg>"
lst_pos_tags = ['NN', 'NNP', 'NNS', 'JJ', 'CD', 'VB', 'VBN', 'VBD', 'VBG', 'RB', 'VBP', 'VBZ', 'NNPS', 'JJS']
stop_words = stopwords.words('english')
isTestRun = False
#Uncomment for a test run
#isTestRun = True

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

tokenizer = AutoTokenizer.from_pretrained(model_pretrained)
if(no_pre):
    eou_token = tokenizer.eos_token
model = RobertaForMaskedLM.from_pretrained(model_pretrained)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)
nlp = spacy.load("en_core_web_sm")
print("Model Loaded")

#Setting log file
log_file = os.path.join(model_dir, 'log.txt')
logging.basicConfig(filename=log_file, filemode='a', 
                    format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logging.Formatter.converter = time.gmtime
logger = logging.getLogger(__name__)

logger.info(f"eou_token={eou_token}, knlg_token={knlg_token}")
logger.info(f"args: {args}")

language = "en"
max_ngram_size = 1
deduplication_threshold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=max_keyword, features=None)

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

def get_pos_tags(doc):
    pos_tags = {}
    for token in doc:
        if(not (token.is_stop or token.is_punct or token.is_space or token.text.lower() in stop_words)):
            if(token.tag_ in lst_pos_tags):
                pos_tags[token.text] = token.tag_
    return pos_tags

def get_mask_words(txt, tok, mapping, add_pos):
    if(mask_random):
        n_sample = math.ceil(0.15*len(mapping))
        mask = random.sample(range(len(mapping)),n_sample)
        mask_words = []
        for idx in mask:
            start, end = tok.word_to_chars(idx)
            word = txt[start:end].lower()
            mask_words.append(word)
    else:
        yake_doc = txt.replace(eou_token, " ")
        yake_doc = yake_doc.strip()
        keywords = custom_kw_extractor.extract_keywords(yake_doc)
        lst_kw = [kw[0] for kw in keywords]

        if(len(lst_kw)<max_keyword and add_pos):
            n = max_keyword-len(lst_kw)
            txt_doc = nlp(txt)
            pos_tags = get_pos_tags(txt_doc)
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
    return mask, mask_words
    
def get_masked_tokens(tokenizer, tok, mapping, mask):
    input_ids = tok["input_ids"].copy()
    labels = [-100]*len(input_ids)
    for word_id in mask:
        for idx in mapping[word_id]:
            labels[idx] = input_ids[idx]
            input_ids[idx] = tokenizer.mask_token_id
    return input_ids, labels

def get_tensors(lst_input, lst_label, lst_attn):
    m = len(lst_input)
    lst_input.extend([tokenizer.pad_token_id]*(max_len-m))
    lst_label.extend([-100]*(max_len-m))
    lst_attn.extend([0]*(max_len-m))
    input_ids = torch.tensor([lst_input], dtype=torch.long)
    labels = torch.tensor([lst_label], dtype=torch.long)
    attn_mask = torch.tensor([lst_attn], dtype=torch.long)
    return input_ids, labels, attn_mask
    
def get_data(txt_input):
    #logger.info(txt_input)
    lst_input_ids = []
    lst_labels = []
    lst_attn_mask = []
    
    utt_list = []
    context = None
    condition = None
    use_condition = False
    
    if(knlg_token in txt_input):
        arr  = txt_input.split(knlg_token)
        context  = arr[0].strip()
        condition = arr[1].strip()
        #logger.info(f"condition: {condition}")
        #logger.info("-"*30)
        use_condition = True
    else:
        context = txt_input
        
    arr = context.split(eou_token)
    utt_list = []
    for i in range(len(arr)-1):
        utt = arr[i].strip()
        utt_list.append(utt)
        #logger.info(utt)
        #logger.info("-"*30)
    
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
        return lst_input_ids, lst_labels, lst_attn_mask
    
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
    
    if(prev is not None or use_condition):
        map_resp = get_word_mapping(tok_resp)
        mask, mask_words = get_mask_words(resp, tok_resp, map_resp, add_pos)
        #logger.info(f"mask_words: {mask_words}")
        
        if(len(mask)>0):
            resp_masked, lbl_resp = get_masked_tokens(tokenizer, tok_resp, map_resp, mask)
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
            id1, label1, attn_mask1 = get_tensors(tok1, lbl1, attn1)
            lst_input_ids.append(id1)
            lst_labels.append(label1)
            lst_attn_mask.append(attn_mask1)
            
            #logger.info(f"id1 : {id1.shape} {id1}")
            #logger.info(f"label1 : {label1.shape} {label1}")
            #logger.info(f"attn_mask1 : {attn_mask1.shape} {attn_mask1}")
            #logger.info("-"*30)
            #logger.info("-"*30)
        
    return lst_input_ids, lst_labels, lst_attn_mask

def prepare_dataset(samples, mode):
    input_ids = []
    attention_masks = []
    labels = []
    for dataset in samples:
        n = 100 if(isTestRun) else len(samples[dataset][mode])
        for i in range(n):
            lst_input_ids, lst_labels, lst_attn_mask = get_data(samples[dataset][mode][i])
            input_ids.extend(lst_input_ids)
            labels.extend(lst_labels)
            attention_masks.extend(lst_attn_mask)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.cat(labels, dim=0)
    
    logger.info(f"Shape of input_ids: {input_ids.shape}")
    logger.info(f"Shape of attention_masks: {attention_masks.shape}")
    logger.info(f"Shape of labels: {labels.shape}")
    
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(
            dataset,
            sampler = RandomSampler(dataset),
            batch_size = BATCH_SIZE
        )
    return dataloader

#----------------------------

samples = get_samples(eou_token, knlg_token, logger)

logger.info("Train Data:-")
train_dataloader = prepare_dataset(samples, "train")
logger.info(f'len(train_dataloader): {len(train_dataloader)}')

logger.info("Validation Data:-")
valid_dataloader = prepare_dataset(samples, "valid")
logger.info(f'len(valid_dataloader): {len(valid_dataloader)}')

logger.info("Test Data:-")
test_dataloader = prepare_dataset(samples, "test")
logger.info(f'len(test_dataloader): {len(test_dataloader)}')

logger.info("Data Loaded")
print("Data Loaded")

#----------------------------

def evaluate_loss(dataloader, model):
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            b_input_ids = batch[0].to(device)
            b_attn_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            output = model(input_ids = b_input_ids, attention_mask = b_attn_mask, labels = b_labels)
            if torch.cuda.device_count() > 1:
                loss = output.loss.mean()
            else:
                loss = output.loss
            total_loss = total_loss + loss.item()
            
    avg_loss = total_loss/len(dataloader)
    return avg_loss

optimizer = AdamW(model.parameters(), lr=learning_rate)
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

tokenizer.save_pretrained(model_dir)
progress_bar = tqdm(range(num_training_steps))
best_valid_loss = 9999
best_epoch = -1
logger.info("-"*40)
for epoch in range(num_train_epochs):
    model.train()
    logger.info("Epoch {} --------------------------".format(epoch+1))
    for i, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_attn_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad(set_to_none=True)
        output = model(input_ids = b_input_ids, attention_mask = b_attn_mask, labels = b_labels)
        loss = output.loss
        if torch.cuda.device_count() > 1:
            loss.mean().backward()
        else:
            loss.backward()
        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)
        
    train_loss = evaluate_loss(train_dataloader, model)
    valid_loss = evaluate_loss(valid_dataloader, model)
    test_loss = evaluate_loss(test_dataloader, model)
    logger.info("Epoch {} : Train loss = {} : Valid loss = {} : Test loss = {}".format(epoch+1, train_loss, valid_loss, test_loss))
    
    if(valid_loss < best_valid_loss):
        best_valid_loss = valid_loss
        best_epoch = epoch+1
        if torch.cuda.device_count() > 1:
            model.module.save_pretrained(model_dir)
        else:
            model.save_pretrained(model_dir)
        logger.info(f"Epoch {epoch+1} model saved.") 
        logger.info("-"*40)
    else:
        logger.info("Early Stopping ...")
        logger.info("-"*40)
        break

logger.info(f"Best Model: {best_epoch}")
logger.info("-"*40)
        
print("done")
#----------------------------