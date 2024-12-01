# EnsMulHateCyb
An Ensemble method

readenglishdata.py
readzhdata.py

讀取原資料集並做前處理、把資料分成label和word，最後再隨機分出訓練和測試集

rdallcsv.py

讀取處理好的csv檔並根據內容生成Embedding

allmodelinone.py

BiLSTM等基礎神經網路的架構

ensallinone.py

訓練主程式

allensPred.py

根據訓練好的模型進行預測並算分數

GloVe english pre-trained word vectors 下載

https://nlp.stanford.edu/projects/glove/

中文詞嵌入向量 TMUNLP_1.6B_WB_300DIM_2020V1.BIN.GZ

https://nlp.tmu.edu.tw/word2vec/index.html
