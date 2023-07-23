import torch
import os, random, math
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, RobertaForMaskedLM
import collections
import numpy as np
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from transformers import default_data_collator
import logging
import argparse
import time
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm
from prepare_dataset_mlm import load_dailydialog, load_personachat, load_wizard_of_wikipedia, load_topical_chat

#----------------------------
"""
Pre-train RoBERTa model on MLM task with different dialogue datasets. 
Utterances are concatenated with a new special token i.e. eou_token (<eou>). 

python train_mlm.py -path=<output_mlm> -epochs=30

"""
#----------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-path','--path', help='path of the model directiory', required=True)
parser.add_argument('-epochs','--epochs', help='training epochs', type=int, required=False, default=30)
args = vars(parser.parse_args())
model_dir = args['path']
epochs = args['epochs']

if(os.path.isdir(model_dir)):
    print("Model Directory exists.")
    exit(0)
else:
    os.mkdir(model_dir)
    print(f"Model directory {model_dir} created.")

#--------------------------------

max_len = 256
batch_size = 64
mlm_probability = 0.15
eou_token = "<eou>"

SEED = 10
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)      
    device = torch.device("cuda")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = RobertaForMaskedLM.from_pretrained("roberta-base")
special_tokens_dict = {'additional_special_tokens': [eou_token]}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)
tokenizer.save_pretrained(model_dir)
print("Model Loaded")

#Setting log file
log_file = os.path.join(model_dir, 'log.txt')
logging.basicConfig(filename=log_file, filemode='a', 
                    format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logging.Formatter.converter = time.gmtime
logger = logging.getLogger(__name__)

#--------------------------------

def tokenize_function(examples):
    result = tokenizer(examples["text"], max_length=max_len, padding='max_length')
    result["labels"] = result["input_ids"].copy()
    return result

def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

def build_dataloader(lm_dataset):

    #Train dataloader
    train_dataloader = DataLoader(
        lm_dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    
    #Eval dataloader
    eval_data = lm_dataset["validation"]
    eval_dataset = eval_data.map(
        insert_random_mask,
        batched=True,
        remove_columns=eval_data.column_names,
    )
    eval_dataset = eval_dataset.rename_columns(
        {
            "masked_input_ids": "input_ids",
            "masked_attention_mask": "attention_mask",
            "masked_labels": "labels",
        }
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
    )
    return train_dataloader, eval_dataloader

def build_dataset(): 
    train_samples = []
    valid_samples = []
    
    #Daily Dialog
    dt_train = load_dailydialog(max_len, tokenizer, eou_token, "train", logger)
    dt_valid = load_dailydialog(max_len, tokenizer, eou_token, "validation", logger)
    train_samples.extend(dt_train)
    valid_samples.extend(dt_valid)
    logger.info(f"train_samples {len(dt_train)} : valid_samples: {len(dt_valid)}")
    logger.info("-"*30)
    
    #PersonaChat
    dt_train = load_personachat(max_len, tokenizer, eou_token, "train", logger)
    dt_valid = load_personachat(max_len, tokenizer, eou_token, "valid", logger)
    train_samples.extend(dt_train)
    valid_samples.extend(dt_valid)
    logger.info(f"train_samples {len(dt_train)} : valid_samples: {len(dt_valid)}")
    logger.info("-"*30)
    
    #Wizard of Wikipedia
    dt_train = load_wizard_of_wikipedia(max_len, tokenizer, eou_token, "train", logger)
    train_samples.extend(dt_train)
    dt_valid1 = load_wizard_of_wikipedia(max_len, tokenizer, eou_token, "valid_random_split", logger)
    dt_valid2 = load_wizard_of_wikipedia(max_len, tokenizer, eou_token, "valid_topic_split", logger)
    valid_samples.extend(dt_valid1)
    valid_samples.extend(dt_valid2)
    
    logger.info(f"train_samples {len(dt_train)} : valid_samples: {len(dt_valid1)+len(dt_valid2)}")
    logger.info("-"*30)
    
    #TopicalChat
    dt_train = load_topical_chat(max_len, tokenizer, eou_token, "train", logger)
    train_samples.extend(dt_train)
    dt_valid1 = load_topical_chat(max_len, tokenizer, eou_token, "valid_rare", logger)
    dt_valid2 = load_topical_chat(max_len, tokenizer, eou_token, "valid_freq", logger)
    valid_samples.extend(dt_valid1)
    valid_samples.extend(dt_valid2)

    logger.info(f"train_samples {len(dt_train)} : valid_samples: {len(dt_valid1)+len(dt_valid2)}")
    logger.info("-"*30)
    
    logger.info(f"train_samples {len(train_samples)} : valid_samples: {len(valid_samples)}")
    
    #Create Dataset
    dt_train = Dataset.from_list(train_samples)
    dt_valid = Dataset.from_list(valid_samples)
    dataset = DatasetDict({"train":dt_train, "validation":dt_valid})
    lm_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    logger.info(lm_dataset)
    train_dataloader, eval_dataloader = build_dataloader(lm_dataset)
    return train_dataloader, eval_dataloader

train_dataloader, eval_dataloader = build_dataset()
logger.info(f"Len train_dataloader: {len(train_dataloader)}")
logger.info(f"Len eval_dataloader: {len(eval_dataloader)}")
    
#--------------------------------

#Set Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_train_epochs = epochs
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

#--------------------------------

def evaluate_loss(dataloader, model):
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            b_input_ids = batch["input_ids"].to(device)
            b_attn_mask = batch["attention_mask"].to(device)
            b_labels = batch["labels"].to(device)

            output = model(input_ids = b_input_ids, attention_mask = b_attn_mask, labels = b_labels)
            if torch.cuda.device_count() > 1:
                loss = output.loss.mean()
            else:
                loss = output.loss
            total_loss = total_loss + loss.item()
            
    avg_loss = total_loss/len(dataloader)
    return avg_loss

progress_bar = tqdm(range(num_training_steps))
best_valid_loss = 9999
best_epoch = -1

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        b_input_ids = batch["input_ids"].to(device)
        b_attn_mask = batch["attention_mask"].to(device)
        b_labels = batch["labels"].to(device)
        
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

    # Evaluation
    train_loss = evaluate_loss(train_dataloader, model)
    valid_loss = evaluate_loss(eval_dataloader, model)
    logger.info("Epoch {} : Train loss = {} : Eval loss = {}".format(epoch+1, train_loss, valid_loss))
    if(valid_loss < best_valid_loss):
        best_valid_loss = valid_loss
        best_epoch = epoch+1
        if torch.cuda.device_count() > 1:
            model.module.save_pretrained(model_dir)
        else:
            model.save_pretrained(model_dir)
        logger.info(f"Epoch {epoch+1} model saved.") 
        logger.info("-"*40)
    
print("done")
    
#--------------------------------