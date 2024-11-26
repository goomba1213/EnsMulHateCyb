import numpy as np
import csv
import math  
import re
import jieba

longest = 50
TAG_RE = re.compile(r'<[^>]+>')
def preprocess_text(sen):
    # Removing html tags
    sentence = TAG_RE.sub('', sen)

    #留中英文
    sentence = re.sub('[^\u4e00-\u9fa5A-Za-z]', ' ', sentence)
    #只留中文
    #sentence = re.sub('[^\u4e00-\u9fa5]', ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence
exclude = ["!","?",","," "]

datasetindex=0
datasetlist=["PTT_2label", "PTT_4label", "IGS_2label", "IGS_4label", "COLDataset"]
csvdatas=["/new_training_data_2label.csv", "/new_testing_data_2label.csv", "/new_training_data_4label.csv", "/new_testing_data_4label.csv", "/new_training_data_2label.csv", "/new_testing_data_2label.csv", "/new_training_data_4label.csv", "/new_testing_data_4label.csv", "/new_train.csv", "/new_test.csv"]

for dataset in datasetlist:
    if dataset == "IGS_2label":
        longest = 10
    else:
        longest = 50

    path = "Dataset/" + dataset
    with open(path + csvdatas[datasetindex], 'r',encoding="utf-8") as f:
        datasetindex+=1
        #alllines = f.readlines()
        reader = csv.reader(f)
        data = list(reader)
    labels = []
    #data = np.empty((len(alllines),0))

    #print(data)
    data_array = np.array(data)
    data_array = np.delete(data_array,0,0)
    data = []
    index = 0

    for line in data_array:
        #print(line)
        if line[2] == "ABU":
            labels.append([1])
        else:
            labels.append([0])
        text = line[1]
        text = preprocess_text(text)
        cutted_text = jieba.cut(text)
        splitedword = []
        #np.append(data,splitedword,axis=0)
        for word in cutted_text:
            if word not in exclude:
                splitedword.append(word)
        #print(splitedword)
        data.append(splitedword)
        index = index + 1

    index=0

    for onedata in data:
        #print(str(onedata) + " " + str(labels[index]))
        labels[index].insert(0," ".join(onedata))
        if len(onedata) < longest:
            for padtime in range(longest - len(onedata)):
                data[index].append("<pad>")
        #print(str(data[index]) + " " + str(labels[index]))
        else:
            data[index] = data[index][:longest]
        index = index + 1
    print(longest)

    data.insert(0,list(range(longest)))
    labels.insert(0,["text","label"])


    with open(path + '/' + dataset +'word_train.csv', 'w', newline='',encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # 寫入二維表格
        writer.writerows(data)
    with open(path + '/' + dataset +'label_train.csv', 'w', newline='',encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(labels)


    with open(path + csvdatas[datasetindex], 'r',encoding="utf-8") as f:
        datasetindex += 1
        #alllines = f.readlines()
        reader = csv.reader(f)
        testdata = list(reader)
    testlabels = []
    #data = np.empty((len(alllines),0))

    #print(testdata)
    test_array = np.array(testdata)
    test_array = np.delete(test_array,0,0)
    testdata = []
    index = 0

    for line in test_array:
        #print(line)
        if line[2] == "ABU":
            testlabels.append([1])
        else:
            testlabels.append([0])
        text = line[1]
        text = preprocess_text(text)
        cutted_text = jieba.cut(text)
        splitedword = []
        #np.append(data,splitedword,axis=0)
        for word in cutted_text:
            if word not in exclude:
                splitedword.append(word)
        #print(splitedword)
        testdata.append(splitedword)
        index = index + 1

    index=0

    for onedata in testdata:
        #print(str(onedata) + " " + str(labels[index]))
        testlabels[index].insert(0," ".join(onedata))
        if len(onedata) < longest:
            for padtime in range(longest - len(onedata)):
                testdata[index].append("<pad>")
        #print(str(data[index]) + " " + str(labels[index]))
        else:
            testdata[index] = testdata[index][:longest]
        index = index + 1
    print(longest)

    testdata.insert(0,list(range(longest)))
    testlabels.insert(0,["text","label"])


    with open(path + '/' + dataset +'word_test.csv', 'w', newline='',encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # 寫入二維表格
        writer.writerows(testdata)
    with open(path + '/' + dataset +'label_test.csv', 'w', newline='',encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(testlabels)

    testdata.pop(0)
    testlabels.pop(0)

    data += testdata
    labels += testlabels
    with open(path + '/' + dataset +'word.csv', 'w', newline='',encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # 寫入二維表格
        writer.writerows(data)
    with open(path + '/' + dataset +'label.csv', 'w', newline='',encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(labels)