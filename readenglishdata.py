import numpy as np
import csv
import math  
import re

longest = 100
TAG_RE = re.compile(r'<[^>]+>')
def preprocess_text(sen):
    # Removing html tags
    sentence = TAG_RE.sub('', sen)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

#Ethos
dataset = "Ethos"
path = "Dataset/" + dataset
with open(path + "\Ethos_Dataset_Binary.csv", 'r',encoding="utf-8") as f:
    alllines = f.readlines()
    #reader = csv.reader(f)
    #data = list(reader)
labels = []
#data = np.empty((len(alllines),0))
data = []
index = 0

for line in alllines:
    print(line)
    splitedline=line.split(";")

    floatlabel = float(splitedline[-1])

    totalsplit = len(splitedline)
    if totalsplit > 2:
        print(splitedline)
        splitedline.pop(-1)
        print(splitedline)
        temp = ""
        for sent in splitedline:
            temp = temp + sent
        text = temp
        #text = str.join(splitedline)
    else:
        text = splitedline[0]

    labels.append([math.ceil(floatlabel)])
    text = text.replace(",",'')
    
    #for sym in toreplace:
        #text = text.replace(sym, '')
    text = preprocess_text(text)
    print(text)
    splitedword=text.split()        
    #np.append(data,splitedword,axis=0)
    data.append(splitedword)
    index = index + 1

index=0

for onedata in data:
    #print(str(onedata) + " " + str(labels[index]))
    labels[index].insert(0," ".join(onedata))
    if len(onedata) < longest:
        size = longest - len(onedata)
        #for padtime in range(longest - len(onedata)):
            #data[index].append("<pad>")
        data[index].extend(["<pad>"]* size)
    else:
        data[index] = data[index][:longest]
    #print(str(data[index]) + " " + str(labels[index]))
    index = index + 1

data.insert(0,list(range(longest)))
labels.insert(0,["text","label"])
print(longest)

with open(path + '/' + dataset +'word.csv', 'w', newline='',encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # 寫入二維表格
    writer.writerows(data)
with open(path + '/' + dataset +'label.csv', 'w', newline='',encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(labels)

import random
#Generate 5 random numbers between 10 and 30
randomlist = random.sample(range(1,len(data)), int(len(data)*3/10))
print(randomlist)
test = [data[0]]
testlab = [labels[0]]
for splitdata in range(len(data)):
    if splitdata in randomlist:
        test.append(data[splitdata])
        testlab.append(labels[splitdata])

with open(path + '/' + dataset +'word_test.csv', 'w', newline='',encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # 寫入二維表格
    writer.writerows(test)
with open(path + '/' + dataset +'label_test.csv', 'w', newline='',encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(testlab)
    
test.pop(0)
testlab.pop(0)
for dele in test:
    data.remove(dele)
for dele in testlab:
    labels.remove(dele)
with open(path + '/' + dataset +'word_train.csv', 'w', newline='',encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # 寫入二維表格
    writer.writerows(data)
with open(path + '/' + dataset +'label_train.csv', 'w', newline='',encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(labels)


#Hatecheck
dataset = "Hatecheck"
path = "Dataset/" + dataset
with open(path + "/all_cases.csv", 'r',encoding="utf-8") as f:
    #alllines = f.readlines()
    reader = csv.reader(f)
    data = list(reader)
labels = []
#data = np.empty((len(alllines),0))

print(data)
data_array = np.array(data)
data_array = np.delete(data_array,0,0)
data = []
index = 0
for line in data_array:
    print(line)
    if line[4] == "hateful":
        labels.append([1])
    else:
        labels.append([0])
    text = line[3]
    text = preprocess_text(text)
    splitedword=text.split()
    #np.append(data,splitedword,axis=0)
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

with open(path + '/' + dataset +'word.csv', 'w', newline='',encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # 寫入二維表格
    writer.writerows(data)
with open(path + '/' + dataset +'label.csv', 'w', newline='',encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(labels)

import random
#Generate 5 random numbers between 10 and 30
randomlist = random.sample(range(1,len(data)), int(len(data)*3/10))
print(randomlist)
test = [data[0]]
testlab = [labels[0]]
for splitdata in range(len(data)):
    if splitdata in randomlist:
        test.append(data[splitdata])
        testlab.append(labels[splitdata])

with open(path + '/' + dataset +'word_test.csv', 'w', newline='',encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # 寫入二維表格
    writer.writerows(test)
with open(path + '/' + dataset +'label_test.csv', 'w', newline='',encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(testlab)

test.pop(0)
testlab.pop(0)
for dele in test:
    data.remove(dele)
for dele in testlab:
    labels.remove(dele)
with open(path + '/' + dataset +'word_train.csv', 'w', newline='',encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # 寫入二維表格
    writer.writerows(data)
with open(path + '/' + dataset +'label_train.csv', 'w', newline='',encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(labels)

#MultiOff
dataset = "MultiOff"
path = "Dataset/" + dataset
with open(path + "/Training_meme_dataset.csv", 'r',encoding="utf-8") as f:
    #alllines = f.readlines()
    reader = csv.reader(f)
    data = list(reader)

with open(path + "/Validation_meme_dataset.csv", 'r',encoding="utf-8") as f:
    #alllines = f.readlines()
    reader = csv.reader(f)
    data = data + list(reader)
labels = []
#data = np.empty((len(alllines),0))

print(data)
data_array = np.array(data)
data_array = np.delete(data_array,0,0)
data = []
index = 0
for line in data_array:
    print(line)
    if line[2] == "offensive":
        labels.append([1])
    else:
        labels.append([0])
    text = line[1]
    text = preprocess_text(text)
    splitedword=text.split()
    #np.append(data,splitedword,axis=0)
    data.append(splitedword)
    index = index + 1

index = 0
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

with open(path + "/Testing_meme_dataset.csv", 'r',encoding="utf-8") as f:
    #alllines = f.readlines()
    reader = csv.reader(f)
    testdata = list(reader)

testlabels = []
#data = np.empty((len(alllines),0))

print(testdata)
test_array = np.array(testdata)
test_array = np.delete(test_array,0,0)
testdata = []
index = 0
for line in test_array:
    print(line)
    if line[2] == "offensive":
        testlabels.append([1])
    else:
        testlabels.append([0])
    text = line[1]
    text = preprocess_text(text)
    splitedword=text.split()
    #np.append(data,splitedword,axis=0)
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

data = data + testdata
labels = labels + testlabels

test.pop(0)
testlab.pop(0)

with open(path + '/' + dataset +'word.csv', 'w', newline='',encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # 寫入二維表格
    writer.writerows(data)
with open(path + '/' + dataset +'label.csv', 'w', newline='',encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(labels)

#Vicos
import os
dataset = "Vicos"
path = "Dataset/" + dataset
with open(path + "/annotations_metadata.csv", 'r',encoding="utf-8") as f:
    #alllines = f.readlines()
    reader = csv.reader(f)
    filedata = list(reader)
    filedata.pop(0)
labels = []
#data = np.empty((len(alllines),0))

train_files = os.listdir(path+"/sampled_train")

data = []
index = 0
for file in train_files:
    with open(path + "/sampled_train/" + file, 'r',encoding="utf-8") as f:
        sent=f.read()
    for filedataindex in filedata:
        if filedataindex[0] == file[:-4]:
            if filedataindex[4] == "hate":
                labels.append([1])
            else:
                labels.append([0])
    
    text = preprocess_text(sent)
    splitedword=text.split()
    #np.append(data,splitedword,axis=0)
    data.append(splitedword)
    index = index + 1

index = 0
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

test_files = os.listdir(path+"/sampled_test")

testlabels = []
#data = np.empty((len(alllines),0))

testdata = []
index = 0

for file in test_files:
    with open(path + "/sampled_test/" + file, 'r',encoding="utf-8") as f:
        sent=f.read()
    for filedataindex in filedata:
        if filedataindex[0] == file[:-4]:
            if filedataindex[4] == "hate":
                testlabels.append([1])
            else:
                testlabels.append([0])
    
    text = preprocess_text(sent)
    splitedword=text.split()
    #np.append(data,splitedword,axis=0)
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

data = data + testdata
labels = labels + testlabels

test.pop(0)
testlab.pop(0)

with open(path + '/' + dataset +'word.csv', 'w', newline='',encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # 寫入二維表格
    writer.writerows(data)
with open(path + '/' + dataset +'label.csv', 'w', newline='',encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(labels)

#========================================================== OffensEval2019 ==========================================================

dataset = "OffensEval2019"
path = "Dataset/" + dataset
with open(path + "/olid-training-v2.0.csv", 'r',encoding="utf-8") as f:
    #alllines = f.readlines()
    reader = csv.reader(f)
    data = list(reader)
labels = []
#data = np.empty((len(alllines),0))

print(data)
data_array = np.array(data)
data_array = np.delete(data_array,0,0)
data = []
index = 0

for line in data_array:
    print(line)
    if line[2] == "OFF":
        labels.append([1])
    else:
        labels.append([0])
    text = line[1]
    text = preprocess_text(text).lower()
    splitedword=text.split()
    #np.append(data,splitedword,axis=0)
    print(splitedword)
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


with open(path + "/testset-levela.csv", 'r',encoding="utf-8") as f:
    #alllines = f.readlines()
    reader = csv.reader(f)
    testdata = list(reader)
testlabels = []
#data = np.empty((len(alllines),0))

print(testdata)
test_array = np.array(testdata)
test_array = np.delete(test_array,0,0)
testdata = []
index = 0

for line in test_array:
    print(line)
    if line[2] == "OFF":
        testlabels.append([1])
    else:
        testlabels.append([0])
    text = line[1]
    text = preprocess_text(text).lower()
    splitedword=text.split()
    #np.append(data,splitedword,axis=0)
    print(splitedword)
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