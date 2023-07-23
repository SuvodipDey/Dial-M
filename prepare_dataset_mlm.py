import os
import json
import re

#Set data path
DAILYDIALOG_PATH = os.path.join("raw_data", "ijcnlp_dailydialog")
PERSONACHAT_PATH = os.path.join("raw_data", "personachat")
WIZARDOFWIKI_PATH = os.path.join("raw_data", "wizard_of_wikipedia")
TOPICALCHAT_PATH = "../Topical-Chat-master/conversations"
MULTIWOZ_PATH = os.path.join("raw_data", "MultiWOZ_2.1")
#--------------

def loadJson(data_file):
    if os.path.isfile(data_file):
        with open(data_file, 'r') as read_file:
            data = json.load(read_file)
            return data
        
def load_list_file(list_file):
    with open(list_file, 'r') as read_file:
        dialog_id_list = read_file.readlines()
        dialog_id_list = [l.strip('\n') for l in dialog_id_list]
        return dialog_id_list
    return

def get_samples(data, max_len, tokenizer, eou_token):
    samples = []
    long_count = 0
    short_count = 0
    min_tok = 32
    tok_max = 0
    
    for utt_list in data:
        i = 0
        cur_idx = 0
        n = len(utt_list)
        context = ""
        tok_count = 0
        while(i<n):
            ids = tokenizer(utt_list[i] + f"{eou_token}")['input_ids']
            tok_count = tok_count + len(ids) - 2
            if(tok_count+2<=max_len):
                context = context + utt_list[i] + f"{eou_token}"
                i+=1
            else:
                c_ids = tokenizer(context.strip())['input_ids']
                if(tok_count>=min_tok and len(c_ids)<=max_len):
                    samples.append({"text": context.strip()})
                    #samples.append(context.strip())
                    if(len(c_ids)>tok_max):
                        tok_max = len(c_ids)
                else:
                    short_count+=1
                context = ""
                tok_count = 0
                if(len(ids)>max_len):
                    long_count+=1
                    break
        
        c_ids = tokenizer(context.strip())['input_ids']
        if(tok_count>=min_tok and len(c_ids)<=max_len):
            samples.append({"text": context.strip()})
            #samples.append(context.strip())
            if(len(c_ids)>tok_max):
                tok_max = len(c_ids)
        else:
            short_count+=1
            
    return samples, short_count, long_count, tok_max

def load_dailydialog(max_len, tokenizer, eou_token, mode, logger):
    samples = []
    file_name = os.path.join(DAILYDIALOG_PATH, mode, f"dialogues_{mode}.txt")
    lines = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
    
    data = []
    v = 0
    for line in lines:            
        arr = line.strip().split('__eou__')
        utt_list = []
        for j in range(len(arr)-1):
            utt_text = arr[j].strip()
            utt_list.append(utt_text)
            v+=1
        data.append(utt_list)
        
    samples, short_count, long_count, tok_max = get_samples(data, max_len, tokenizer, eou_token)
    logger.info(f"DailyDialog {mode}:: {len(lines)} {v}:: Long texts: {long_count}: Short texts: {short_count}: tok_max: {tok_max}")
    return samples

def load_personachat(max_len, tokenizer, eou_token, mode, logger):
    file_name = os.path.join(PERSONACHAT_PATH, f"{mode}_both_original.txt")
    with open(file_name, 'r') as f:
        lines = f.readlines()
        persona_data = {}
        c = -1
        v = 0
        for line in lines:
            if("1 your persona:" in line):
                c+=1
                persona_data[c] = {}
                persona_data[c]['p_self_spk1'] = []
                persona_data[c]['p_partner_spk0'] = []
                persona_data[c]['dialog'] = []
            if(" persona:" in line):
                p = line.split(" persona:")[1]
                if(" your persona:" in line):
                    persona_data[c]['p_self_spk1'].append(p.strip())
                else:
                    persona_data[c]['p_partner_spk0'].append(p.strip())
            else:
                p = line.split("\t")
                c1 = p[0].strip()
                c1 = re.sub("^(\d)+ ", "", c1)
                c2 = p[1].strip()
                persona_data[c]['dialog'].append(c1)
                persona_data[c]['dialog'].append(c2)
                v+=2
                
    data = []
    for k in persona_data:
        data.append(persona_data[k]['dialog'])
        
    samples, short_count, long_count, tok_max = get_samples(data, max_len, tokenizer, eou_token)
    logger.info(f"PersonaChat {mode}:: {len(data)} {v}:: Long texts: {long_count}: Short texts: {short_count}: tok_max: {tok_max}")
    return samples   

def load_wizard_of_wikipedia(max_len, tokenizer, eou_token, mode, logger):
    path = os.path.join(WIZARDOFWIKI_PATH, f"{mode}.json")
    data = []
    v = 0
    with open(path, 'r') as f:
        wow_data = json.load(f)
        
    for k in wow_data:
        dialog = k['dialog']
        utt_list = []
        for i in range(len(dialog)):
            utt_list.append(dialog[i]['text'].strip())
            v+=1
        data.append(utt_list)
            
    samples, short_count, long_count, tok_max = get_samples(data, max_len, tokenizer, eou_token)
    logger.info(f"Wizard-of-Wikipedia {mode}:: {len(wow_data)} {v}:: Long texts: {long_count}: Short texts: {short_count}: tok_max: {tok_max}")
    return samples 

def load_topical_chat(max_len, tokenizer, eou_token, mode, logger):
    path = os.path.join(TOPICALCHAT_PATH, f"{mode}.json")
    data = []
    v = 0
    with open(path, 'r') as f:
        topical_data = json.load(f)
        
    for key in topical_data:
        dialog = topical_data[key]['content']
        utt_list = []
        for i in range(len(dialog)):
            utt_list.append(dialog[i]['message'].strip())
            v+=1
        data.append(utt_list)
            
    samples, short_count, long_count, tok_max = get_samples(data, max_len, tokenizer, eou_token)
    logger.info(f"Topical-Chat {mode}:: {len(topical_data)} {v}:: Long texts: {long_count}: Short texts: {short_count}: tok_max: {tok_max}")
    return samples 

def load_multiwoz(max_len, tokenizer, eou_token, mode, logger):
    dialog_data_file = os.path.join(MULTIWOZ_PATH, 'data.json')
    dialog_data = loadJson(dialog_data_file)
    dialog_id_list = list(set(dialog_data.keys()))
    valid_list_file = os.path.join(MULTIWOZ_PATH, 'valListFile.txt')
    test_list_file = os.path.join(MULTIWOZ_PATH, 'testListFile.txt')
    valid_id_list = list(set(load_list_file(valid_list_file)))
    test_id_list = load_list_file(test_list_file)
    train_id_list = [did for did in dialog_id_list if did not in (valid_id_list + test_id_list)]
    
    if(mode=="valid"):
        dt = [(k,v) for k, v in dialog_data.items() if k in valid_id_list]
    elif(mode=="test"):
        dt = [(k,v) for k, v in dialog_data.items() if k in test_id_list]
    else:
        dt = [(k,v) for k, v in dialog_data.items() if k in train_id_list]
    
    data = []
    v = 0
    for k,d in dt:
        utt_list = []
        for i, t in enumerate(d['log']):
            v+=1
            utt_text = t['text'].strip().replace("\n", "")
            utt_text = utt_text.replace("\t", "")
            utt_text = utt_text.replace("’","'")
            utt_text = re.sub("(\s*)([a-zA-Z|0-9]+)(\.| \.|\. )([a-zA-Z]+)(\s*)", r"\1\2 . \4\5", utt_text)
            utt_list.append(utt_text.strip())
        data.append(utt_list)

    samples, short_count, long_count, tok_max = get_samples(data, max_len, tokenizer, eou_token)
    logger.info(f"MultiWOZ {mode}:: {len(data)} {v}:: Long texts: {long_count}: Short texts: {short_count}: tok_max: {tok_max}")
    return samples
