import os
import torch
from transformers import AutoModel, AutoTokenizer,AdamW,get_linear_schedule_with_warmup
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import random
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score       
from taskB_triplet_dataset import Dataset       
import torch.nn.functional as F 
from triplet_classification import Classifier
import argparse

parser=argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--emb_size',type=int,default=768)

args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
def test_fn(test_dataloader,model, device):

    model.eval()

    acc=0
    true_label=[]
    preds=[]
    with torch.no_grad():
        for batch in test_dataloader:
            ids=batch['input_ids'].to(device)
            xmsk=batch['attention_mask'].to(device)
            label=batch['label'].to(device)
        
            output=model(ids,xmsk)
            
    
            pred=torch.argmax(F.softmax(output),axis=1)
            #print(pred, label)
            preds.extend(pred.detach().cpu().numpy())
           
            true_label.extend(label.detach().cpu().numpy())
        
        acc=accuracy_score(true_label,preds)
        precision=precision_score(true_label,preds, average='weighted')
        recall=recall_score(true_label,preds,average='weighted')
        macro=f1_score(true_label,preds,average='macro')   
        micro=f1_score(true_label,preds,average='micro')  
        return acc,precision,recall,macro,micro
            
        
if __name__=="__main__":
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    
    model_name='sentence-transformers/paraphrase-distilroberta-base-v1' 
    embed=AutoModel.from_pretrained(model_name)
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    
    EMB_MODEL_PATH='/home/labuser/Semeval/Triplet/model/16_1e-05_1.9286644317685289.pt'
    embed.load_state_dict(torch.load(EMB_MODEL_PATH, map_location=device)['model_state_dict'])

    
    TEST_PATH="/home/labuser/Semeval/Data/SubtaskB/subtaskB_test.jsonl"


    testdf=pd.read_json(TEST_PATH,lines=True)
    test_data=testdf['text'].tolist()
    test_label=testdf['label'].values
    test_dataset=Dataset(test_data,test_label,tokenizer)
    

    batch_size=args.batch_size
    n_labels=6
    seed=42
    embed_size=args.emb_size
    set_seed(seed)
    
    test_dataloader=DataLoader(test_dataset,shuffle=False,batch_size=batch_size)

    
    
    MODEL_PATH="/home/labuser/Semeval/Triplet/classification/32_1e-05_0.7.pt"
    model = Classifier(embed,embed_size,n_labels).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'])
    test_accuracy,precision,recall,test_macro,test_micro=test_fn(test_dataloader,model, device)
    print(f"test_accuracy : {test_accuracy},precision : {precision},recall: {recall},f1 : {test_macro,test_micro}")