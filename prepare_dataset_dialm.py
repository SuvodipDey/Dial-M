import os
import json
import re

#Set data path
DAILYDIALOG_PATH = os.path.join("raw_data", "ijcnlp_dailydialog")
PERSONACHAT_PATH = os.path.join("raw_data", "personachat")
WIZARDOFWIKI_PATH = os.path.join("raw_data", "wizard_of_wikipedia")
TOPICALCHAT_PATH = "../Topical-Chat-master/TopicalChatEnriched"
TOPICALCHAT_KN_PATH = os.path.join("../Topical-Chat-master", "reading_sets", "post-build")
MULTIWOZ_PATH = os.path.join("raw_data", "MultiWOZ_2.1")
#--------------
domain_set = {'hospital', 'restaurant', 'attraction', 'hotel', 'police', 'booking', 'train', 'taxi'}
slot_dict = {'area': 'area', 'post': 'postal code', 'depart': 'departure location', 'name': 'name', 'type': 'type', 'ref': 'reference number', 'choice': 'choice', 'price': 'price range', 'arrive': 'arrival time', 'internet': 'internet', 'food': 'food type', 'leave': 'leaving time', 'parking': 'parking', 'department': 'department', 'day': 'day of booking', 'car': 'car type', 'stay': 'length of stay', 'open': 'open', 'time': 'time', 'dest': 'destination', 'stars': 'stars', 'people': 'people', 'id': 'id', 'phone': 'phone', 'fee': 'fee', 'ticket': 'ticket price', 'addr': 'address'}
#--------------

def convert_system_act(sys_act):
    txt = ""
    for act in sys_act:
        arr = act.split("-")
        domain = arr[0].strip()
        if(domain.lower() not in domain_set):
            continue
        act_type = arr[1]                
        if(act_type.lower()=="request"):
            val = []
            for sv in sys_act[act]:
                if(sv[0].lower() not in ["none", "choice"]):
                    val.append(slot_dict[sv[0].lower()])
            if(len(val)>0):
                str_val = ", ".join(val)
                s = f" Request for {domain.lower()} {str_val}."
                txt = txt + s
        else:
            val = []
            for sv in sys_act[act]:
                if(sv[0].lower() not in ["none", "choice"]):
                    if(sv[0].lower() in ['internet', 'parking']):
                        h = ""
                        if(sv[1].lower()=="none"):
                            h = "free "
                        elif(sv[1].lower()=="no"):
                            h = "no "
                        val.append(f"{h}{slot_dict[sv[0].lower()]}")
                    else:
                        val.append(f"{slot_dict[sv[0].lower()]} {sv[1]}")
            if(len(val)>0):
                str_val = ", ".join(val)
                if(act_type.lower() in ["nobook", "nooffer"]):
                    s = f" {domain} not available with {str_val}."
                elif(act_type.lower() in ["book", "offerbook", "offerbooked"]):
                    if(domain.lower()=="booking"):
                        s = f" Booked with {str_val}."
                    else:
                        s = f" {domain} booked with {str_val}."
                elif(act_type.lower()=="select"):
                    s = f" Ask to select {domain.lower()} from {str_val}."
                else:
                    s = f" {act_type} {domain.lower()} with {str_val}."
                txt = txt + s   
    return txt.strip()

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

def load_dailydialog(eou_token, mode, logger):
    file_name = os.path.join(DAILYDIALOG_PATH, mode, f"dialogues_{mode}.txt")
    lines = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines: 
        context = ""
        arr = line.strip().split('__eou__')
        for j in range(len(arr)-1):
            utt_text = arr[j].strip()
            utt_text = utt_text.replace("’","'")
            utt_text = re.sub("(\s*)([a-zA-Z|0-9]+)(\.| \.|\. )([a-zA-Z]+)(\s*)", r"\1\2 . \4\5", utt_text)
            context = context + utt_text.strip() + f"{eou_token}"
            if(j>0):
                utt = context.strip()
                data.append(utt)
        
    logger.info(f"DailyDialog {mode}:: {len(lines)} {len(data)}")
    return data

def load_personachat(eou_token, knlg_token, mode, logger):
    file_name = os.path.join(PERSONACHAT_PATH, f"{mode}_both_original.txt")
    with open(file_name, 'r') as f:
        lines = f.readlines()
    persona_data = {}
    c = -1
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
                
    data = []
    for k in persona_data:
        dialog = persona_data[k]['dialog']
        p_spk0 = " ".join(persona_data[k]['p_partner_spk0'])
        p_spk1 = " ".join(persona_data[k]['p_self_spk1'])
        
        context = ""
        for i in range(len(dialog)):
            utt_text = dialog[i]
            utt_text = utt_text.replace("’","'")
            utt_text = re.sub("(\s*)([a-zA-Z|0-9]+)(\.| \.|\. )([a-zA-Z]+)(\s*)", r"\1\2 . \4\5", utt_text)
            context = context + utt_text.strip() + f"{eou_token}"
            if(i>0):
                persona = p_spk0 if (i%2==0) else p_spk1
                utt = context.strip() + f"{knlg_token}" + persona.strip()
                data.append(utt)
        
    logger.info(f"PersonaChat {mode}:: {len(persona_data)} {len(data)}")
    return data   

def load_wizard_of_wikipedia(eou_token, knlg_token, mode, logger):
    path = os.path.join(WIZARDOFWIKI_PATH, f"{mode}.json")
    data = []
    v = 0
    with open(path, 'r') as f:
        wow_data = json.load(f)
        
    data = []
    for k in wow_data:
        dialog = k['dialog']
        context = ""
        for i in range(len(dialog)):
            utt_text = dialog[i]['text'].strip()
            utt_text = utt_text.replace("’","'")
            utt_text = re.sub("(\s*)([a-zA-Z|0-9]+)(\.| \.|\. )([a-zA-Z]+)(\s*)", r"\1\2 . \4\5", utt_text)
            context = context + utt_text.strip() + f"{eou_token}"
            if "Wizard" in dialog[i]["speaker"]:
                utt = context.strip()
                if ("no_passages_used" in dialog[i]['checked_sentence'] or 
                    len(dialog[i]['checked_sentence']) == 0):
                    data.append(utt)
                else:
                    for t in dialog[i]['checked_sentence']:
                        utt = utt + f"{knlg_token}" + dialog[i]['checked_sentence'][t].strip()
                        data.append(utt)
                        break
            
    logger.info(f"Wizard-of-Wikipedia {mode}:: {len(wow_data)} {len(data)}")
    return data 

def load_topical_chat(eou_token, knlg_token, mode, logger):
    path = os.path.join(TOPICALCHAT_PATH, f"{mode}.json")
    data = []
    with open(path, 'r') as f:
        topical_data = json.load(f)
    
    knowledge_path = os.path.join(TOPICALCHAT_KN_PATH, f"{mode}.json")
    with open(knowledge_path, 'r') as f:
        knowledge = json.load(f)

    for key in topical_data:
        dialog = topical_data[key]['content']
        context = ""
        for i in range(len(dialog)):
            utt_text = " ".join(dialog[i]['message'])
            utt_text = utt_text.replace("’","'")
            utt_text = re.sub("(\s*)([a-zA-Z|0-9]+)(\.| \.|\. )([a-zA-Z]+)(\s*)", r"\1\2 . \4\5", utt_text)
            context = context + utt_text.strip() + f"{eou_token}"
            utt = context.strip()
            agent = dialog[i]['agent']
            try:
                if dialog[i]['gt_turn_ks']['section'][0] == 'F':
                    if dialog[i]['gt_turn_ks']['ds'] == 'wiki':
                        start = dialog[i]['gt_turn_ks']['start_index']
                        end = dialog[i]['gt_turn_ks']['end_index']
                        knlg = knowledge[key][agent][dialog[i]['gt_turn_ks']['section']]['shortened_wiki_lead_section'][start:end+1]
                        utt = utt + f"{knlg_token}" + knlg.strip()
                        data.append(utt)
                    elif dialog[i]['gt_turn_ks']['ds'] == 'fun_facts':
                        knlg = knowledge[key][agent][dialog[i]['gt_turn_ks']['section']]['fun_facts'][dialog[i]['gt_turn_ks']['index']]
                        utt = utt + f"{knlg_token}" + knlg.strip()
                        data.append(utt)
                else:
                    start = dialog[i]['gt_turn_ks']['start_index']
                    end = dialog[i]['gt_turn_ks']['end_index']
                    knlg = knowledge[key]['article'][dialog[i]['gt_turn_ks']['section']][start:end+1]
                    utt = utt + f"{knlg_token}" + knlg.strip()
                    data.append(utt)
            except:
                pass
            
    logger.info(f"Topical-Chat {mode}:: {len(topical_data)} {len(data)}")
    return data 

def load_multiwoz(eou_token, knlg_token, dt, mode, logger):
    data = []
    for k,d in dt:
        context = ""
        for i, t in enumerate(d['log']):
            utt_text = t['text'].strip().replace("\n", "")
            utt_text = utt_text.replace("\t", "")
            utt_text = utt_text.replace("’","'")
            utt_text = re.sub("(\s*)([a-zA-Z|0-9]+)(\.| \.|\. )([a-zA-Z]+)(\s*)", r"\1\2 . \4\5", utt_text)
            
            context = context + utt_text.strip() + f"{eou_token}"
            if(i%2==1 and 'dialog_act' in t):
                sys_act = t['dialog_act']
                sys_act_txt = convert_system_act(sys_act)
                utt = context.strip()
                if(sys_act_txt!=""):
                    utt = utt + f"{knlg_token}" + sys_act_txt.strip()
                data.append(utt)

    logger.info(f"MultiWOZ {mode}:: {len(dt)} {len(data)}")
    return data

def get_samples(eou_token, knlg_token, logger): 
    samples = {}
    
    #Daily Dialog
    dt_train = load_dailydialog(eou_token, "train", logger)
    dt_valid = load_dailydialog(eou_token, "validation", logger)
    dt_test = load_dailydialog(eou_token, "test", logger)
    dd_data = {}
    dd_data["train"] = dt_train
    dd_data["valid"] = dt_valid
    dd_data["test"] = dt_test
    samples["dailydialog"] = dd_data
    logger.info(f"train_samples {len(dt_train)} : valid_samples: {len(dt_valid)} : test_samples: {len(dt_test)}")
    logger.info("-"*30)
    
    #PersonaChat
    dt_train = load_personachat(eou_token, knlg_token, "train", logger)
    dt_valid = load_personachat(eou_token, knlg_token, "valid", logger)
    dt_test = load_personachat(eou_token, knlg_token, "test", logger)
    persona_data = {}
    persona_data["train"] = dt_train
    persona_data["valid"] = dt_valid
    persona_data["test"] = dt_test
    samples["personachat"] = persona_data
    logger.info(f"train_samples {len(dt_train)} : valid_samples: {len(dt_valid)} : test_samples: {len(dt_test)}")
    logger.info("-"*30)
    
    #Wizard of Wikipedia
    dt_train = load_wizard_of_wikipedia(eou_token, knlg_token, "train", logger)
    dt_valid = load_wizard_of_wikipedia(eou_token, knlg_token, "valid_random_split", logger)
    dt_valid.extend(load_wizard_of_wikipedia(eou_token, knlg_token, "valid_topic_split", logger))
    dt_test = load_wizard_of_wikipedia(eou_token, knlg_token, "test_random_split", logger)
    dt_test.extend(load_wizard_of_wikipedia(eou_token, knlg_token, "test_topic_split", logger))
    woz_data = {}
    woz_data["train"] = dt_train
    woz_data["valid"] = dt_valid
    woz_data["test"] = dt_test
    samples["woz"] = woz_data
    logger.info(f"train_samples {len(dt_train)} : valid_samples: {len(dt_valid)} : test_samples: {len(dt_test)}")
    logger.info("-"*30)
    
    #TopicalChat
    dt_train = load_topical_chat(eou_token, knlg_token, "train", logger)
    dt_valid = load_topical_chat(eou_token, knlg_token, "valid_rare", logger)
    dt_valid.extend(load_topical_chat(eou_token, knlg_token, "valid_freq", logger))
    dt_test = load_topical_chat(eou_token, knlg_token, "test_rare", logger)
    dt_test.extend(load_topical_chat(eou_token, knlg_token, "test_freq", logger))
    topical_data = {}
    topical_data["train"] = dt_train
    topical_data["valid"] = dt_valid
    topical_data["test"] = dt_test
    samples["topical"] = topical_data
    logger.info(f"train_samples {len(dt_train)} : valid_samples: {len(dt_valid)} : test_samples: {len(dt_test)}")
    logger.info("-"*30)
    
    #MultiWOZ
    system_act_file = os.path.join(MULTIWOZ_PATH, 'system_acts.json')
    system_acts = loadJson(system_act_file)
    dialog_data_file = os.path.join(MULTIWOZ_PATH, 'data.json')
    dialog_data = loadJson(dialog_data_file)
    dialog_id_list = list(set(dialog_data.keys()))
    valid_list_file = os.path.join(MULTIWOZ_PATH, 'valListFile.txt')
    test_list_file = os.path.join(MULTIWOZ_PATH, 'testListFile.txt')
    valid_id_list = list(set(load_list_file(valid_list_file)))
    test_id_list = load_list_file(test_list_file)
    train_id_list = [did for did in dialog_id_list if did not in (valid_id_list + test_id_list)]
    train_data = [(k,v) for k, v in dialog_data.items() if k in train_id_list]
    valid_data = [(k,v) for k, v in dialog_data.items() if k in valid_id_list]
    test_data = [(k,v) for k, v in dialog_data.items() if k in test_id_list]
    
    dt_train = load_multiwoz(eou_token, knlg_token, train_data, "train", logger)
    dt_valid = load_multiwoz(eou_token, knlg_token, valid_data, "valid", logger)
    dt_test = load_multiwoz(eou_token, knlg_token, test_data, "test", logger)
    dd_data = {}
    dd_data["train"] = dt_train
    dd_data["valid"] = dt_valid
    dd_data["test"] = dt_test
    samples["multiwoz"] = dd_data
    logger.info(f"train_samples {len(dt_train)} : valid_samples: {len(dt_valid)} : test_samples: {len(dt_test)}")
    logger.info("-"*30)
    
    return samples
