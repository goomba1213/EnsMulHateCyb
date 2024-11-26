import numpy as np
import csv

#readdataset = ["Ethos", "Hatecheck", "MultiOff", "Vicos", "OffensEval2019"] 
#readdataset = ["PTT_2label", "IGS_2label", "PTT_4label", "IGS_4label", "COLDataset"]
readdataset = ["Ethos"]
outlang = "eng"

array = []
for dataset in readdataset:
    path = "Dataset\\"+ dataset +"\\"+dataset+"word_train.csv"
    with open(path, 'r',encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)

    data.pop(0)
    array += data
    print(len(array))


with open("Embeddings/" + outlang + '_' + '_'.join(readdataset) + "word2vec.txt","w",encoding="utf-8") as wr:
    for sentence in array:
        for word in sentence:
            if word != '<pad>':
                wr.write(word+" ")
            else:
                wr.write("<pad>\n")
                break

import multiprocessing
word_dim_size =300
import gensim
from gensim.models.word2vec import Word2Vec
#model = gensim.models.KeyedVectors.load_word2vec_format('詞嵌入模型檔的路徑', unicode_errors='ignore', binary=True)
from gensim.models import word2vec

max_cpu_counts = multiprocessing.cpu_count()
word_dim_size = 300  #  設定 word vector 維度
usesent = "Embeddings/" + outlang +  '_' + '_'.join(readdataset) + "word2vec.txt"
print(f"Use {max_cpu_counts} workers to train Word2Vec (dim={word_dim_size})")

sentences = word2vec.LineSentence(usesent)
for sentence in sentences:
    print(sentence)
#sentences = word2vec.LineSentence(usesent)

# 訓練模型
#IGSmodel = word2vec.Word2Vec(sentences, vector_size=word_dim_size, workers=max_cpu_counts)
#IGSmodel = word2vec.Word2Vec.load('word2vec_IGS_300_model')
#PTTmodel = word2vec.Word2Vec.load('word2vec_PTT_300_model')
#Twittermodel = word2vec.Word2Vec.load('word2vec_Twitter_300_model')

Allmodel = word2vec.Word2Vec(sentences, vector_size=word_dim_size, workers=max_cpu_counts, min_count=5)

print(Allmodel)
print(Allmodel.wv.vectors.shape)
output_model = "Embeddings/" + outlang + '_' + '_'.join(readdataset) + "_dim"+ str(word_dim_size) +"_model.w2v"
Allmodel.save(output_model)