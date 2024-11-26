import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

embed_size = 300
import gensim
from gensim.models.word2vec import Word2Vec
#model = gensim.models.KeyedVectors.load_word2vec_format('詞嵌入模型檔的路徑', unicode_errors='ignore', binary=True)
from gensim.models import word2vec
#model = word2vec.Word2Vec.load('zh_en_COMB_dim300_model')
"""
readdataset = ["MultiOff"]
trainlanguage = "eng"

use_train_dataset = ["Ethos","Hatecheck","MultiOff","Vicos","OffensEval2019"]
#readdataset = ["Ethos","Hatecheck","MultiOff","Vicos"]
"""

readdataset = ["Ethos"]
#trainlanguage = "chn"
trainlanguage = "eng"
use_train_dataset = ["Ethos"]
#readdataset = ["PTT_2label", "IGS_2label", "PTT_4label", "IGS_4label", "COLDataset"]

toload = "Embeddings/" + trainlanguage + '_' + '_'.join(use_train_dataset) + "_dim"+ str(embed_size) +"_model.w2v"
model = word2vec.Word2Vec.load(toload)

vocab = model.wv.index_to_key
vocab_size =len(vocab)
word_to_idx = {word: i+1 for i, word in enumerate(vocab)}
idx_to_word = {i+1: word for i, word in enumerate(vocab)}

def encode_samples(tokenized_samples): #use word index mapping to encode token
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in word_to_idx:
                feature.append(word_to_idx[token])
            else:
                #print(token)
                feature.append(word_to_idx["<pad>"])
        #print(feature)
        features.append(feature)
    return features


import torch
import numpy as np
import csv

index = 0
for toread in readdataset:
    with open("Dataset\\" + toread + "\\" + toread + "word_train.csv", 'r',encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)

    data_array = np.array(data)
    data_array = np.delete(data_array,0,0)
    seq_len = len(data_array[0])
    with open("Dataset\\" + toread + "\\" + toread + "word_test.csv", 'r',encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)

    test_array = np.array(data)
    test_array = np.delete(test_array,0,0)

    with open("Dataset\\" + toread + "\\" + toread + "label_train.csv", 'r',encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)

    train_label_arr = []
    fir = 1
    for a in data:
        if fir != 1:
            train_label_arr.append(int(a[1]))
        else:
            fir = 0

    with open("Dataset\\" + toread + "\\" + toread + "label_test.csv", 'r',encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)

    test_label_arr = []
    fir = 1
    for a in data:
        if fir != 1:
            test_label_arr.append(int(a[1]))
        else:
            fir = 0
    if index == 0:
        total_data_array = data_array
        total_test_array = test_array
        total_train_label_arr = train_label_arr
        total_test_label_arr = test_label_arr
    else:
        total_data_array = np.append(total_data_array,data_array,axis=0)
        total_test_array = np.append(total_test_array,test_array,axis=0)
        total_train_label_arr = np.append(total_train_label_arr,train_label_arr,axis=0)
        total_test_label_arr = np.append(total_test_label_arr,test_label_arr,axis=0)
    #print(total_data_array)
    index+=1

### 將token轉成index 並轉成 pytorch tensor ###

train_features = torch.tensor(encode_samples(total_data_array))
train_labels = torch.tensor(total_train_label_arr)

test_features = torch.tensor(encode_samples(total_test_array))
test_labels = torch.tensor(total_test_label_arr)

### create pytorch dataloader ###
batch_size = 32
train_set = torch.utils.data.TensorDataset(train_features, train_labels)

#分成訓練與驗證
train_size = int(0.8 * len(train_set))
val_size = len(train_set) - train_size
print(train_size,val_size)
train_set, val_set = torch.utils.data.random_split(train_set, [train_size, val_size])

train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True)
#val_set = torch.utils.data.TensorDataset(val_features, val_labels)
val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size,shuffle=False)

test_set = torch.utils.data.TensorDataset(test_features, test_labels)
test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict(net,test_iter):
    #state = torch.load(os.path.join(cwd,'checkpoint','epoch10_maxlen300_embed200.pt'),map_location=torch.device('cpu'))
    #net.load_state_dict(state['state_dict'])
    pred_list = []
    true_list = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        net.eval()
        for batch,label in test_iter:
            output = net(batch.to(device))
            pred_list.extend(torch.argmax(softmax(output),dim=1).cpu().numpy())
            true_list.extend(label.cpu().numpy())

    acc = accuracy_score(pred_list, true_list)
    pre = precision_score(pred_list, true_list)
    reca = recall_score(pred_list, true_list)
    f1 = f1_score(pred_list, true_list)
    #print(pred_list)
    #print(true_list)
    TP=0
    TN=0
    FP=0
    FN=0
    Pos=0
    Neg=0
    for predindex in range(len(pred_list)):
        #print("pred : "+str(pred_list[predindex])+" true : "+str(true_list[predindex]))
        if str(true_list[predindex]) == "1":
            Pos+=1
        else:
            Neg+=1
        if pred_list[predindex] == true_list[predindex]:
            if str(pred_list[predindex]) == "1":
                TP+=1
            else:
                TN+=1
        else:
            if str(true_list[predindex]) == "1":
                FN+=1
            else:
                FP+=1
        
    print('test acc: %f'%acc)
    print('test precision: %f'%pre)
    print('test recall: %f'%reca)
    print('test f1: %f'%f1)
    print("TP: " + str(TP) + " TN: " + str(TN) + " FP: " + str(FP) + " FN: " + str(FN))
    print("Pos: " + str(Pos) + " Neg: " + str(Neg))
    return acc,pred_list,true_list

import torch.nn as nn
import torch.nn.functional as F
from allmodelinone import BiLSTM, BiGRU, CNNLSTM

#宣告集成網路
class Ens(nn.Module):
    def __init__(self, modellstm, modelgru, modelcnnlstm):
        super(Ens, self).__init__()
        self.modellstm = modellstm
        self.modelgru = modelgru
        self.modelcnnlstm = modelcnnlstm
        self.classifier = nn.Linear(6,2)
        num_hiddens = 100
        #self.classifier = nn.Linear(num_hiddens*20,2)

    def forward(self, xo):
        x1 = self.modellstm(xo)
        x2 = self.modelgru(xo)
        x3 = self.modelcnnlstm(xo)
        x = torch.cat((x1, x2, x3), dim=1)
        #print(x.shape)
        x = self.classifier(F.relu(x))

        return x
    
#readdataset = ["Ethos","Hatecheck","MultiOff","Vicos"]

model_namelist = ["ensemble"]
model_list=[]

for modeln in model_namelist:
    model_list.append(torch.load('model_pts/test_model_'+ modeln + '_' + '_'.join(use_train_dataset) +'.pt'))

model_list[-1].to(device)

acc,pred_list,true_list = predict(model_list[-1],test_iter)