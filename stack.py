## make the final submission file
import pandas as pd
import numpy as np

train=pd.DataFrame()
train['decom']=pd.read_csv('stack_data/decom.csv')['decom']
train['esim']=pd.read_csv('stack_data/esim.csv')['esim']
train['label']=pd.read_csv('stack_data/esim.csv')['label']
print(train.head())
y=train['label'].as_matrix()
train.pop('label')
x=train.as_matrix()
from sklearn.cross_validation import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=3845)
import xgboost as xgb
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.05
params['max_depth'] = 6
d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
bst = xgb.train(params, d_train, 2000, watchlist, early_stopping_rounds=50, verbose_eval=2)

decom_predict=[]
with open("sub/decom_average_data_aug.txt", "r") as f:
    for predict in f:
        decom_predict.append(float(predict))
f.close()
esim_predict=[]
with open("sub/esim_average_data_aug.txt", "r") as f:
    for predict in f:
        esim_predict.append(float(predict))
f.close()
x_test=pd.DataFrame()
x_test['decom']=decom_predict
x_test['esim']=esim_predict
print(x_test.head())
x_test=x_test.as_matrix()
d_test = xgb.DMatrix(x_test)
result=bst.predict(d_test)
with open("sub/stack_13.txt", "w") as f:
    for predict in result:
       f.write(str(predict))
       f.write('\n')


