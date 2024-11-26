import os
import re
import time
from itertools import chain

embed_size = 300
import gensim
from gensim.models.word2vec import Word2Vec
#model = gensim.models.KeyedVectors.load_word2vec_format('詞嵌入模型檔的路徑', unicode_errors='ignore', binary=True)
from gensim.models import word2vec
#model = word2vec.Word2Vec.load('zh_en_COMB_dim300_model')

#readdataset = ["Ethos", "Hatecheck", "MultiOff", "Vicos", "OffensEval2019"]
trainlanguage = "eng"
#readdataset = ["PTT_2label", "IGS_2label", "PTT_4label", "IGS_4label", "COLDataset"]
readdataset = ["Ethos"]
#trainlanguage = "chn"

toload = "Embeddings/" + trainlanguage + '_' + '_'.join(readdataset) + "_dim"+ str(embed_size) +"_model.w2v"
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
"""
if "IGS_4label" in readdataset:
    hatelineindex = []
    for train_index in range(len(total_train_label_arr)):
        if total_train_label_arr[train_index] == 1:
            hatelineindex.append(train_index)
    
    hatelines = []
    hatelabels = []
    for hate_index in hatelineindex:
        hatelines.append(total_data_array[hate_index])
        hatelabels.append(total_train_label_arr[hate_index])
    for apptime in range(7):
        total_data_array = np.append(total_data_array,hatelines,axis=0)
        total_train_label_arr = np.append(total_train_label_arr,hatelabels,axis=0)
"""
"""
if "IGS_4label" in readdataset:
    hatelineindex = []
    for train_index in range(len(total_train_label_arr)):
        if total_train_label_arr[train_index] == 0:
            hatelineindex.append(train_index)
    import random
    #產生非仇恨言論總數 7/8 個隨機數
    randomlist = random.sample(range(1,len(hatelineindex)), int(len(hatelineindex)*7/8))
    print(randomlist)
    total_data_array = np.delete(total_data_array,randomlist,axis=0)
    total_train_label_arr = np.delete(total_train_label_arr,randomlist,axis=0)
"""
"""
if "OffensEval2019" in readdataset:
    hatelineindex = []
    for train_index in range(len(total_train_label_arr)):
        if total_train_label_arr[train_index] == 1:
            hatelineindex.append(train_index)
    
    hatelines = []
    hatelabels = []
    for hate_index in hatelineindex:
        hatelines.append(total_data_array[hate_index])
        hatelabels.append(total_train_label_arr[hate_index])
    for apptime in range(1):
        total_data_array = np.append(total_data_array,hatelines,axis=0)
        total_train_label_arr = np.append(total_train_label_arr,hatelabels,axis=0)
"""
hates=0
for train_index in range(len(total_train_label_arr)):
    if total_train_label_arr[train_index] == 1:
        hates+=1
print(hates)

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


### load word2vec model ###
#pre-train model download from: https://github.com/stanfordnlp/GloVe
#preprocess:https://stackoverflow.com/questions/51323344/cant-load-glove-6b-300d-txt
#wvmodel = gensim.models.KeyedVectors.load_word2vec_format('y_360W_cbow_2D_300dim_2020v1.bin', unicode_errors='ignore', binary=True)
if trainlanguage == "eng":
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format('glove6B/glove.6B.50d.txt', unicode_errors='ignore', no_header=True, encoding='utf-8')
    embed_size = 50
else:
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format('y_360W_cbow_2D_300dim_2020v1.bin', unicode_errors='ignore', binary=True)
    embed_size = 300
#wvmodel = gensim.models.KeyedVectors.load_word2vec_format("nonuseembeddings\\vectors_en.txt", unicode_errors='ignore', binary=False, limit=2519370)

#wvmodel = model
# map golve pretrain weight to pytorch embedding pretrain weight

weight = torch.zeros(vocab_size+1, embed_size) #given 0 if the word is not in glove
use=0
for i in range(len(wvmodel.index_to_key)):
    try:
        index = word_to_idx[wvmodel.index_to_key[i]] #transfer to our word2ind
        use+=1
    except:
        continue
    #weight[index, :] = torch.from_numpy(wvmodel.get_vector(wvmodel.index_to_key[i]))
    vec = wvmodel.get_vector(wvmodel.index_to_key[i])
    vec.setflags(write=1)
    weight[index, :] = torch.from_numpy(vec)
print(use)

"""
#用自己的embedding權重
weight = torch.zeros(vocab_size+1, embed_size) #given 0 if the word is not in glove
for i in range(vocab_size):
    weight[i, :] = torch.from_numpy(model.wv.get_vector(model.wv.index_to_key[i]))
"""
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

num_epochs = 10
num_hiddens = 100
num_layers = 2
bidirectional = True
labels = 2
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#建立3種網路
model_bilstm = BiLSTM(vocab_size=(vocab_size+1), embed_size=embed_size,
                   num_hiddens=num_hiddens, num_layers=num_layers,
                   bidirectional=bidirectional, weight=weight,
                   labels=labels)
model_bigru = BiGRU(vocab_size=(vocab_size+1), embed_size=embed_size,
                   num_hiddens=num_hiddens, num_layers=num_layers,
                   bidirectional=bidirectional, weight=weight,
                   labels=labels)

model_cnnlstm = CNNLSTM(vocab_size=(vocab_size+1), embed_size=embed_size, seq_len=seq_len, labels=labels, weight=weight, num_hiddens=num_hiddens, num_layers=num_layers, bidirectional=bidirectional,)
"""
#分別讀取訓練好的模型
model_bilstm = torch.load("test_lstm.pt")
model_bigru = torch.load("test_gru.pt")
model_cnnlstm = torch.load("test_cnnlstm.pt")
"""
model_ensemble = Ens(model_bilstm, model_bigru, model_cnnlstm)

model_list = [model_bilstm, model_bigru, model_cnnlstm, model_ensemble]
model_namelist = ["bilstm", "bigru", "cnnlstm", "ensemble"]

import torch.autograd as autograd
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

#建立集成網路
model_ensemble = Ens(model_bilstm, model_bigru, model_cnnlstm)

def train(net, num_epochs, loss_function, optimizer, train_iter, val_iter, modelnum):
    for epoch in range(num_epochs):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0
        net.train()
        for feature, label in train_iter:
            n += 1
            optimizer.zero_grad()
            feature = Variable(feature.to(device))
            label = Variable(label.to(device)).to(torch.int64)

            score = net(feature)
            loss = loss_function(score, label)
            loss.backward()
            optimizer.step()
            train_acc += accuracy_score(torch.argmax(score.cpu().data,dim=1), label.cpu())
            train_loss += loss

        with torch.no_grad():
            net.eval()
            for val_feature, val_label in val_iter:
                m += 1
                val_feature = val_feature.to(device)
                val_label = val_label.to(device).to(torch.int64)
                val_score = net(val_feature)
                val_loss = loss_function(val_score, val_label)
                val_acc += accuracy_score(torch.argmax(val_score.cpu().data,dim=1), val_label.cpu())
                val_losses += val_loss

        runtime = time.time() - start
        print('epoch: %d, train loss: %.4f, train acc: %.2f, val loss: %.4f, val acc: %.2f, time: %.2f' %
              (epoch, train_loss.data/n, train_acc/n, val_losses.data/m, val_acc/m, runtime))

    #save final model
    state = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
            }
    #torch.save(state,'last_model.pt')
    torch.save(net,'model_pts/test_model_'+ model_namelist[modelnum] + '_' + '_'.join(readdataset) +'.pt')


def predict(net, test_iter, modelnum):
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

for modelnum in range(4):

    loss_function = nn.CrossEntropyLoss() # ~ nn.LogSoftmax()+nn.NLLLoss()
    optimizer = torch.optim.Adam(model_list[modelnum].parameters())

    if modelnum == 3:
        for param in model_list[modelnum].parameters():
            param.requires_grad = False

        for param in model_list[modelnum].classifier.parameters():
            param.requires_grad = True    

    model_list[modelnum].to(device)

    print('start to train model' + model_namelist[modelnum])
    train(model_list[modelnum], num_epochs, loss_function, optimizer, train_iter, val_iter, modelnum)

    print('start to predict test set ' + model_namelist[modelnum])
    acc,pred_list,true_list = predict(model_list[modelnum], test_iter, modelnum)

print('Done')
