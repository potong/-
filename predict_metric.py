
## metric models
import pandas as pd
import numpy as np
def metric(y_true,y_pred):
    log_loss=0
    num=len(y_true)
    pos=0
    neg=0
    pos_correct=0
    pos_wrong=0
    neg_correct=0
    neg_wrong=0
    for i in range(num):
        label=y_true[i]
        pred=y_pred[i]
        if label>0.5:
            log_loss-=np.log(pred+0.0000001)
            pos+=1
            if pred>0.5:
                pos_correct+=1
            else:  pos_wrong+=1
        else:
            log_loss-=np.log(1-pred+0.0000001)
            neg+=1
            if pred >0.5:
                neg_wrong+=1
            else:
                neg_correct+=1
    print("log loss is :",log_loss/num)
    print("total num is ",num," correct is :",pos_correct+neg_correct," wrong is ",pos_wrong+neg_wrong,
          "rate is " ,(pos_correct+neg_correct)/num)
    print("pos number is :",pos, "  correct is :",pos_correct,"  wrong is :",pos_wrong,
          "rate is ",pos_correct/pos)
    print("neg number is :", neg, "  correct is :", neg_correct, "  wrong is :",neg_wrong,
          "rate is ", neg_correct / neg)

import os
documents=os.listdir('stack_data')
for document in documents:
    print("document :",document)
    if document!='test.csv':
        path='stack_data/'+document
        train=pd.read_csv(path)
        y = train['label'].as_matrix()
        train.pop('label')
        columns = list(train.columns)
        for column in columns:
            print("the model is :", column)
            y_pred = train[column].as_matrix()
            metric(y, y_pred)

