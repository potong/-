import pandas as pd
from itertools import combinations
import numpy as np
import re
spanlish_train_path="ori_data/cikm_spanish_train_20180516.txt"
english_train_path="ori_data/cikm_english_train_20180516.txt"
f = open(english_train_path, encoding='utf-8')## 20000
span1=[]
span2=[]
label=[]
for line in f:
    line = line.lower()
    sentences = re.sub('[?,()".¿:*!/\n_|;&#}@{]', ' ', line)
    sentences=re.sub('$€', ' euro', sentences)
    sentences = re.split(r'[\t]', sentences)
    span1.append(re.sub('[0123456789]', '   ', sentences[1]))
    span2.append(re.sub('[0123456789]', '  ', sentences[3]))
    label.append(int(sentences[4]))
f.close()
train_data = pd.DataFrame()
train_data['span1'] = span1
train_data['span2'] = span2
train_data['label'] = label
train_data.to_csv("new_data/train2.csv",index=False)
















train1=pd.read_csv('new_data/train2.csv')
combine_result=[]
def combine_sentence(temp):
    result=[]
    lenth=len(temp)
    if lenth<2:
        return 0
    l = np.arange(lenth)
    combines=list(combinations(l, 2))
    for combine in combines:
        index1=combine[0]
        index2=combine[1]
        result.append([[temp[index1][0],temp[index2][0]],[temp[index1][1],temp[index2][1]]])
    return result
def aug_span1():
    df = train1.sort_values(by=['span1'])
    span1 = list(df['span1'])
    span2 = list(df['span2'])
    label = list(df['label'])
    lenth = len(span1)
    temp=[]
    temp.append([span2[0],label[0]])
    for i in range(1,lenth):
        if span1[i]==span1[i-1]:
            temp.append([span2[i],label[i]])
        else:
            temp=combine_sentence(temp)
            if temp!=0:
               for value in temp:
                   combine_result.append(value)
            temp=[]

def aug_span2():
    df = train1.sort_values(by=['span2'])
    span1 = list(df['span1'])
    span2 = list(df['span2'])
    label = list(df['label'])
    lenth = len(span1)
    temp=[]
    temp.append([span1[0],label[0]])
    for i in range(1,lenth):
        if span2[i]==span2[i-1]:
            temp.append([span1[i],label[i]])
        else:
            temp=combine_sentence(temp)
            if temp!=0:
               for value in temp:
                   combine_result.append(value)
            temp=[]

aug_span1()
aug_span2()
print(len(combine_result))
train_augmentation=pd.DataFrame()
span1=[]
span2=[]
label=[]
for value in combine_result:
    if value[1][0]+value[1][1]==1:
        span1.append(value[0][0])
        span2.append(value[0][1])
        label.append(0)
    elif value[1][0]+value[1][1]==2:
        span1.append(value[0][0])
        span2.append(value[0][1])
        label.append(1)
train_augmentation['span1']=span1
train_augmentation['span2']=span2
train_augmentation['label']=label
train_augmentation.to_csv('new_data/train2_add.csv',index=False)






