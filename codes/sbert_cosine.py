import numpy as np
import os
from transformers import AutoTokenizer,AutoModel,RobertaForSequenceClassification,RobertaTokenizer,RobertaModel
from sentence_transformers import SentenceTransformer, util
import json
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score,accuracy_score,precision_score,recall_score
from sklearn.neighbors import KNeighborsClassifier
import pynndescent
from sklearn.utils import shuffle


def cos_classification(data,label,index,model):
    emb=model.encode(data)
    neighbors=index.query(emb,k=6)
    result=[]
    for i,cos in enumerate(neighbors[1]):
        result.append(neighbors[0][i][0])
    return result

def pooling_embedding(train_embedding,train_label):
    avg_train_embedding={'embedding':[],'label':[]}
    for i in range(6):
        idx=np.where(train_label==i)
        avg_train_embedding['embedding'].append(torch.mean(torch.Tensor(train_embedding[idx]),dim=0).numpy())
        avg_train_embedding['label'].append(i)
    return avg_train_embedding



if __name__=="__main__":
    
    
    train_df=pd.read_json("/home/labuser/Semeval/Data/SubtaskB/subtaskB_train.jsonl",lines=True)
    valid_df=pd.read_json("/home/labuser/Semeval/Data/SubtaskB/subtaskB_dev.jsonl",lines=True)
    test_df=pd.read_json("/home/labuser/Semeval/Data/SubtaskB/subtaskB_test.jsonl",lines=True)

    train_data=train_df['text'].tolist()
    valid_data=valid_df['text'].tolist()
    test_data=test_df['text'].tolist()

    train_label=train_df['label'].values
    valid_label=valid_df['label'].values
    test_label=test_df['label'].values

    train_data,train_label=shuffle(train_data,train_label,random_state=42)
    valid_data,valid_label=shuffle(valid_data,valid_label,random_state=42)


    model_name='paraphrase-distilroberta-base-v1' 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=SentenceTransformer(model_name,device=device)
    
    train_embedding=model.encode(train_data)
    avg_embedding=pooling_embedding(train_embedding,train_label)
    index=pynndescent.NNDescent(torch.Tensor(np.array(avg_embedding['embedding'])),metric='cosine',n_neighbors=100)
    index.prepare()
    
    print("train")
    train_result=cos_classification(train_data,train_label,index,model)
    print("valid")
    valid_result=cos_classification(valid_data,valid_label,index,model)
    print("test")
    test_result=cos_classification(test_data,test_label,index,model)
    
    
    results=[train_result,valid_result,test_result]
    labels=[train_label,valid_label,test_label]
    file="/home/labuser/Semeval/cosine_based/sbert_cos_result.txt"
    for label,result in zip(results,labels):
        
        if os.path.exists(file):
            with open(file,'a') as f:
                print("yes")
                f.write(f"f1_macro={f1_score(label,result,average='macro',zero_division=0)}\n")
                f.write(f"f1_macro={f1_score(label,result,average='micro',zero_division=0)}\n")
                f.write(f"f1_macro={f1_score(label,result,average='weighted',zero_division=0)}\n")
                f.write(f"precision={precision_score(label,result,average='weighted')}\n")
                f.write(f"recall={recall_score(label,result,average='weighted')}\n")
                f.write(f"accuracy={accuracy_score(label,result)}\n")
        else:
            with open(file,'w') as f:
                print("no")
                f.write(f"f1_macro={f1_score(label,result,average='macro',zero_division=0)}\n")
                f.write(f"f1_macro={f1_score(label,result,average='micro',zero_division=0)}\n")
                f.write(f"f1_macro={f1_score(label,result,average='weighted',zero_division=0)}\n")
                f.write(f"precision={precision_score(label,result,average='weighted')}\n")
                f.write(f"recall={recall_score(label,result,average='weighted')}\n")
                f.write(f"accuracy={accuracy_score(label,result)}\n")
    f.close()