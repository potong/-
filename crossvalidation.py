import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from dl_main import *
from data_utils import *
import random
random.seed(18320)
# data load
X1,X2,Y=load_processed_train(MAX_LEN)
#shuffle the data
# index = [i for i in range(len(X1))]
# random.shuffle(index)
# X1=X1[index]
# X2=X2[index]
# Y=Y[index]

def five_split(i,data):
    ## i <5
    ## data is a list include x,y
    data_train=[]
    data_valid=[]
    lenth=len(data[0])
    per_lenth=int(lenth/5)
    if i ==4:
        for data_ in data:
            data_train.append(data_[:4*per_lenth])
            data_valid.append(data_[4*per_lenth:])
    elif i==0:
        for data_ in data:
            data_train.append(data_[per_lenth:])
            data_valid.append(data_[0:per_lenth])
    else:
        for data_ in data:
            data_train.append(np.concatenate([data_[0:i*per_lenth],data_[(i+1)*per_lenth:]]))
            data_valid.append(data_[i*per_lenth:(i+1)*per_lenth])
    return data_train, data_valid
def cross_decom():
    for i in range(5):
        train_data, valid_data = five_split(i, [X1, X2,Y])
        train_x1 = train_data[0]
        train_x2 = train_data[1]
        train_y = train_data[2]


        valid_x1 = valid_data[0]
        valid_x2 = valid_data[1]
        print("stack {a} decom model :".format(a=i))
        decomosable_train(train_x1, train_x2, train_y,i)
        result = decom_predict(valid_x1, valid_x2,i)
        if i == 0:
            decom_stack = result
        else:
            decom_stack = np.concatenate([decom_stack, result])
        test_x1, test_x2 = load_processed_test(MAX_LEN)
        result = decom_predict(test_x1, test_x2,i)
        name = "sub/decom" + str(i) + '.txt'
        with open(name, "w") as f:
            for predict in result:
                f.write(str(predict[0]))
                f.write('\n')
    train = pd.DataFrame()
    train['decom'] = decom_stack[:,0]
    train['label'] = Y
    train.to_csv('stack_data/decom.csv', index=False)
def cross_esim():
    for i in range(5):
        train_data, valid_data = five_split(i, [X1, X2, Y])
        train_x1 = train_data[0]
        train_x2 = train_data[1]
        train_y = train_data[2]
        valid_x1 = valid_data[0]
        valid_x2 = valid_data[1]
        print("stack {a} esim model :".format(a=i))
        esim_train(train_x1, train_x2, train_y,i)
        result = esim_predict(valid_x1, valid_x2,i)
        if i == 0:
            esim_stack = result
        else:
            esim_stack = np.concatenate([esim_stack, result])
        test_x1, test_x2= load_processed_test(MAX_LEN)
        result = esim_predict(test_x1, test_x2,i)
        name="sub/esim"+str(i)+'.txt'
        with open(name, "w") as f:
            for predict in result:
                f.write(str(predict[0]))
                f.write('\n')
    train = pd.DataFrame()
    train['esim'] = esim_stack[:, 0]
    train['label'] = Y[:,0]
    train.to_csv('stack_data/esim.csv', index=False)

def cross_combine():
    for i in range(5):
        train_data, valid_data = five_split(i, [X1, X2, Y])
        train_x1 = train_data[0]
        train_x2 = train_data[1]
        train_y = train_data[2]
        valid_x1 = valid_data[0]
        valid_x2 = valid_data[1]
        print("stack {a} combine model :".format(a=i))
        combine_train(train_x1, train_x2, train_y)
        result = combine_predict(valid_x1, valid_x2)
        if i == 0:
            combine_stack = result
        else:
            combine_stack = np.concatenate([combine_stack, result])
        test_x1, test_x2= load_processed_test(MAX_LEN)
        result = combine_predict(test_x1, test_x2)
        name="sub/combine"+str(i)+'.txt'
        with open(name, "w") as f:
            for predict in result:
                f.write(str(predict[0]))
                f.write('\n')
    train = pd.DataFrame()
    train['combine'] = combine_stack[:, 0]
    train['label'] = Y[:,0]
    train.to_csv('stack_data/combine.csv', index=False)
cross_decom()
# cross_esim()
# cross_combine()
