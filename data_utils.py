import re
import pandas as pd
import numpy as np
spanlish_train_path="ori_data/cikm_spanish_train_20180516.txt"
english_train_path="ori_data/cikm_english_train_20180516.txt"
en_vec_path="word_vec/wiki.en.vec"
span_vec_path="word_vec/wiki.es.vec"

test_path="ori_data/cikm_test_a_20180516.txt"    ## last day change the test a  path to test b
words_dict_path='word_vec/words_dict.txt'
unlabel_path='ori_data/cikm_unlabel_spanish_train_20180516.txt'

words_count_path='word_vec/word_count.txt'
def get_train_data():
    span1 = []
    span2 = []
    label = []
    f = open(spanlish_train_path, encoding='utf-8')    ##1400
    for line in f:
        line=line.lower()
        sentences = re.sub('[?,()".¿:*!/\n_|;&#}@{]', ' ', line)
        sentences=re.sub('$€', ' euro ', sentences)
        sentences = re.split(r'[\t]', sentences)
        span1.append(re.sub('[0123456789]', '   ', sentences[0]))
        span2.append(re.sub('[0123456789]', '  ', sentences[2]))
        label.append(int(sentences[4]))
    f.close()
    f = open(english_train_path, encoding='utf-8')## 20000
    for line in f:
        line = line.lower()
        sentences = re.sub('[?,()".¿:*!/\n_|;&#}@{]', ' ', line)
        sentences=re.sub('$€', ' euro ', sentences)
        sentences = re.split(r'[\t]', sentences)
        span1.append(re.sub('[0123456789]', '  ', sentences[1]))
        span2.append(re.sub('[0123456789]', '  ', sentences[3]))
        label.append(int(sentences[4]))
    f.close()
    train_data = pd.DataFrame()
    train_data['span1'] = span1
    train_data['span1'] = train_data['span1'].str.split(' ')
    train_data['span2'] = span2
    train_data['span2'] = train_data['span2'].str.split(' ')
    train_data['label'] = label
    return train_data

def get_train_sentence():
    span1 = []
    span2 = []
    label = []
    f = open(spanlish_train_path, encoding='utf-8')    ##1400
    for line in f:
        line=line.lower()
        sentences = re.sub('[?,()".¿:*!/\n_|;&#}@{]', ' ', line)
        sentences=re.sub('$€', ' euro', sentences)
        sentences = re.split(r'[\t]', sentences)
        span1.append(re.sub('[0123456789]', '   ', sentences[0]))
        span2.append(re.sub('[0123456789]', '  ', sentences[2]))

        label.append(int(sentences[4]))
    f.close()
    f = open(english_train_path, encoding='utf-8')## 20000
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
    # train_data['span1'] = train_data['span1'].str.split(' ')
    train_data['span2'] = span2
    # train_data['span2'] = train_data['span2'].str.split(' ')
    train_data['label'] = label
    return train_data

def get_test_data():
    span1 = []
    span2 = []
    f = open(test_path, encoding='utf-8')
    for line in f:
        line = line.lower()
        sentences = re.sub('[?,()".¿:*!/\n_|;&#}@{]', ' ', line)
        sentences=re.sub('$€', ' euro ', sentences)
        sentences = re.sub('[0123456789]', '  ', sentences)
        sentences = re.split(r'[\t]', sentences)


        span1.append(sentences[0])
        span2.append(sentences[1])
    f.close()
    test_data = pd.DataFrame()
    test_data['span1'] = span1
    test_data['span1'] = test_data['span1'].str.split(' ')
    test_data['span2'] = span2
    test_data['span2'] = test_data['span2'].str.split(' ')
    return test_data

def get_test_sentence():
    span1 = []
    span2 = []
    f = open(test_path, encoding='utf-8')
    for line in f:
        line = line.lower()
        sentences = re.sub('[?,()".¿:*!/\n_|;&#}@{]', ' ', line)
        sentences=re.sub('$€', ' euro', sentences)
        sentences = re.sub('[0123456789]', '  ', sentences)
        sentences = re.split(r'[\t]', sentences)
        span1.append(sentences[0])
        span2.append(sentences[1])
    f.close()
    test_data = pd.DataFrame()
    test_data['span1'] = span1
    # test_data['span1'] = test_data['span1'].str.split(' ')
    test_data['span2'] = span2
    # test_data['span2'] = test_data['span2'].str.split(' ')
    return test_data

def get_unlabeled_data():
    span1 = []
    span2 = []
    f = open(unlabel_path, encoding='utf-8')
    for line in f:
        line = line.lower()
        sentences = re.sub('[?,()".¿:*!/\n_|;&#}@{]', ' ', line)
        sentences=re.sub('$€', ' euro ', sentences)
        sentences = re.sub('[0123456789]', '  ', sentences)

        sentences = re.split(r'[\t]', sentences)
        span1.append(sentences[0])
    f.close()
    train_data = pd.DataFrame()
    train_data['span1'] = span1
    train_data['span1'] = train_data['span1'].str.split(' ')
    return train_data

def get_span_vec():
    f = open(span_vec_path, encoding='utf-8')
    span_vec={}
    for line in f:

        line=line.split(' ')
        span_vec[line[0]]= [float(f) for f in line[1:-1]]
    return span_vec

def save_dict(path_name,dict_name):        ##save a dict to a txt file ,if dic_test={'a':[1,2,3]}\
                                            ##save_dict('new_data/dic_test.txt',dic_test)
    f = open(path_name, 'w', encoding='utf-8')
    f.write(str(dict_name))
    f.close()
def load_dict(path_name):                    ## eg :dic_test=load_dict('new_data/dic_test.txt')
    f = open(path_name, 'r', encoding='utf-8')
    a = f.read()
    dict_ = eval(a)
    f.close()
    return dict_
def make_word_dict():
    words=list(set(np.hstack(get_test_data()['span1'].as_matrix())))\
               +list(set(np.hstack(get_test_data()['span2'].as_matrix())))\
                +list(set(np.hstack(get_train_data()['span1'].as_matrix())))\
               +list(set(np.hstack(get_train_data()['span2'].as_matrix())))\
                +list(set(np.hstack(get_unlabeled_data()['span1'].as_matrix())))
    words=list(set(words))                              ## all spanish words
    num_of_words=len(words)
    print("num of words: ",num_of_words)
    values=np.arange(num_of_words)                      ##values =[0,1,2 ...]
    words_dict=dict(zip(words,values))
    save_dict(words_dict_path,words_dict)
    print(len(words))
from collections import Counter
def make_counts():
    words = list(np.hstack(get_test_data()['span1'].as_matrix())) \
            + list(np.hstack(get_test_data()['span2'].as_matrix())) \
            + list(np.hstack(get_train_data()['span1'].as_matrix())) \
            + list(np.hstack(get_train_data()['span2'].as_matrix()))\
            + list(np.hstack(get_unlabeled_data()['span1'].as_matrix()))
    counts=Counter(words)
    print("num of words :",len(counts))
    save_dict(words_count_path, counts)

def make_embeding_matrix():
    word_dict=load_dict(words_dict_path)
    word_dict = {value: key for key, value in word_dict.items()}           ## change the values and keys
    span_vec=get_span_vec()                                                ## get the fast_text pretrained vector
    exceed_matrix= np.float32(np.random.uniform(-0.6, 0.6, [2000,300]))   ## if words is not in fast_text

    j=0
    matrix=[]
    num_words = len(word_dict)
    values = np.arange(num_words)

    exceed_word=[]
    for value in values:                             ##values=[0,1,2,3,..]
        word=word_dict[value]
        try :
            vector=span_vec[word]
        except KeyError:
            vector=exceed_matrix[j]
            j+=1
            exceed_word.append(word)
        matrix.append(vector)
    print("no vector words :",exceed_word)
    print("num of exceed words :",j)
    matrix=np.array(matrix)

    matrix[0]=matrix[0]*0.0                    ## mask 0,words of index 0  has vector [0,0,0,...]
    np.save('word_vec/word_enc.npy',matrix)



def pop_list(l,value):  ##delete the element of a list
  j=0
  for i in range(len(l)):
    if l[i-j] == value:
        l.pop(i-j)
        j+=1

  return l

def reverse(list1):   ##change the order
    lenth=len(list1)
    list2=[]
    for i in range(lenth):
        list2.append(list1[lenth-i-1])
    return list2

def hand_length(list1,l2,word_dict):   ## turn the txt list to  number list with a lenth
                                ## list1 is the list ,l2 is the lenth
    # list1=reverse(list1)
    list1=pop_list(list1,'')
    # list1=pop_stop(list1)
    # list1=pop_low_use(list1)
    list2=[]


    list3=[]
    for word in list1:
        try:
            list3.append(word_dict[word])
        except KeyError:
            print("there is a new word :",word)
    l1 = len(list3)
    for i in range(l2):                         ## zero is padding to the end
        if i <l1:
                list2.append(list3[i])
        else:
                list2.append(0)

    return list2


def split_1(row):
    result=str(row['span1']).lower().split()
    return result
def split_2(row):
    result=str(row['span2']).lower().split()
    return result

def load_processed_train(max_lenth):  ## get the structured  train data  like a sentence is 'i like banana '
                                        ## return the [0 0 0 0 0 0 0 34 56 43] if the maxlen=10  and if 35,56,43 is the
                                        # index of i,like banana
    X1=[]
    X2=[]
    Y=[]
    # feature=[]
    train = get_train_data()



    ## data augmentation
    train1=pd.read_csv('new_data/train1.csv')
    train2=pd.read_csv('new_data/train2.csv')
    train1_add=pd.read_csv('new_data/train1_add.csv')
    train2_add=pd.read_csv('new_data/train2_add.csv')
    train=pd.concat([train1,train2,train1_add,train2_add]).reset_index(drop=True)
    train['span1'] = train.apply(split_1, axis=1, raw=True)
    train['span2'] = train.apply(split_2, axis=1, raw=True)
    ## data augmentation

    train = train.sort_values(by=['span1']).reset_index(drop=True)  ##


    span1 = train['span1']
    span2 = train['span2']
    ### can add features extract code to there
    label = list(train['label'])
    word_dict = load_dict(words_dict_path)
    for i in range(len(train)):
        x1 = span1[i]
        x2 = span2[i]
        X1.append(hand_length(list(x1), max_lenth, word_dict))
        X2.append(hand_length(list(x2), max_lenth, word_dict))
        Y.append([label[i]])
    return np.array(X1),np.array(X2),np.array(Y)
def load_processed_test(max_lenth):   ## get the structured  train data
    X1=[]
    X2=[]
    test = get_test_data()
    span1 = test['span1']
    span2 = test['span2']
    word_dict = load_dict(words_dict_path)
    for i in range(len(test)):
         x1=span1[i]
         x2=span2[i]
         X1.append(hand_length(list(x1),max_lenth,word_dict))
         X2.append(hand_length(list(x2),max_lenth,word_dict))
    return np.array(X1),np.array(X2)


