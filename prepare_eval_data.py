import os
import json
import re
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
import csv
from pathlib import Path
from transformers import AutoTokenizer

## Evaluation data path
TC_EVAL = "evaluation_data/tc_usr_data.json"
PC_EVAL = "evaluation_data/pc_usr_data.json"
ENGAGE_EVAL = "evaluation_data/engage_all.csv"
HOLISTIC_EVAL = "evaluation_data/holistic_eval/context_data_release.csv"

def get_format(txt):
    txt = txt.replace("\n", "")
    pred_tokens = word_tokenize(txt.strip().lower())
    s = ' '.join(pred_tokens)
    s = s.replace(" s ", "s ")
    s = s.replace(" nt ", "nt ")
    s = s.replace(" m ", "m ")
    s = s.replace(" - ", "-")
    s = s.replace(" / ", "/")
    s = s.replace("`", "'")
    s = re.sub("(\s*)([a-zA-Z|0-9]+)(\.| \.|\. )([a-zA-Z]+)(\s*)", r"\1\2 . \4\5", s)
    return s

def load_tc_usr(eou_token, knlg_token):
    f = open (TC_EVAL, "r")
    tc_eval = json.loads(f.read())
    f.close()
    
    lst_tc = []
    c = 0
    for dt in tc_eval:
        dct = {}
        context = dt['context']
        knlg = dt['fact']
        responses = dt['responses']
        arr = context.split(" \n ")
        prev = arr[-1]
        utt_prev = get_format(prev)
        resp = responses[0]['response']
        resp = resp.replace("\n","")
        utt_resp = get_format(resp)
        
        dct['prev'] = utt_prev
        dct['resp'] = utt_resp
        dct['idx'] = c
        dct['responses'] = responses
        
        ctx = eou_token.join(arr)+eou_token+resp.strip()+eou_token+knlg_token+knlg.strip()
        ctx = ctx.replace("\n","")
        dct['ctx'] = ctx
        lst_tc.append(dct)
        c+=1
    
    return lst_tc

def load_pc_usr(eou_token, knlg_token):
    f = open (PC_EVAL, "r")
    pc_eval = json.loads(f.read())
    f.close()

    lst_pc = []
    c = 0
    for dt in pc_eval:
        dct = {}
        context = dt['context']
        knlg = dt['fact']
        knlg = knlg.replace("\n","")
        responses = dt['responses']
        resp = responses[0]['response']
        utt_resp = resp.replace("\n","")

        arr = context.split("\n")[:-1]
        arr_knlg = knlg.split("your persona:")[1:]
        persona = "".join(arr_knlg)
        ctx = eou_token.join(arr)+eou_token+utt_resp.strip()+eou_token+knlg_token+persona.strip()
        ctx = ctx.replace("\n","")
        
        dct['context'] = context
        dct['ref'] = utt_resp
        dct['fact'] = knlg
        dct['responses'] = responses
        dct['ctx'] = ctx
        lst_pc.append(dct)
        c+=1
    
    return lst_pc

def load_engage_data(eou_token):
    data = []
    with open(ENGAGE_EVAL, mode ='r') as f:  
        reader = csv.DictReader(f, delimiter=',')
        rows = []
        for row in reader:
            rows.append(row)
        rows = sorted(rows, key=lambda x: x['query'])

        for i in range(0, len(rows), 3):

            context = rows[i]['query'].strip()
            assert len(set([x['query'] for x in rows[i:i+3]])) == 1

            response = rows[i]['response'].strip()

            score = np.mean([float(x['human_score']) for x in rows[i:i+3]])

            if context == '':
                context = 'NA'
            if response == '':
                response = 'NA'

            dct = {}
            dct = {}
            dct['ctx'] = context.strip()+eou_token+response.strip()+eou_token
            dct['Overall'] = score
            data.append(dct)
    return data

def load_holistic_data(eou_token):
    contexts, responses, references, scores = [], [], [], []
    with open(HOLISTIC_EVAL, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            context = row[1] 
            response = row[2]
            human_scores = [int(x) for x in row[3:]]
            score = np.mean(human_scores)

            contexts.append([context])
            responses.append(response)
            references.append('NO REF')
            scores.append(score)

    data = []
    for i in range(len(contexts)):
        ctx = eou_token.join(contexts[i]) + eou_token + responses[i] + eou_token
        dct = {}
        dct['ctx'] = ctx
        dct['Overall'] = scores[i]
        data.append(dct)
    return data
