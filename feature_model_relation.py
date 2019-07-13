## set the deep learning network structure
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv('feature/features.csv')
sub=pd.read_csv('stack_data/decom.csv')
df['decom']=sub['decom']
sub=pd.read_csv('stack_data/mvlstm.csv')
df['esim']=sub['mvlstm']


def accuracy(row):
    s1=row['label']
    s2=row['decom']
    if (s1-0.5)*(s2-0.5)>0:
       return 1
    else: return 0
df['decom']=df.apply(accuracy,axis=1,raw=True)

def accuracy1(row):
    s1=row['label']
    s2=row['esim']
    if (s1-0.5)*(s2-0.5)>0:
       return 1
    else: return 0
df['esim']=df.apply(accuracy1,axis=1,raw=True)
columns=list(df.columns)
for value in ['label','decom','esim']:
    columns.remove(value)
for column in columns:
    df = df.sort_values(by=[column]).reset_index(drop=True)
    pos_rate=pd.rolling_mean(df[['decom']],min_periods=1,window = 600).as_matrix()
    pos_rate=pos_rate[:,0]
    pos_rate1 = pd.rolling_mean(df[['esim']], min_periods=1, window=600).as_matrix()
    pos_rate1 = pos_rate1[:, 0]
    pos_rate2 = pd.rolling_mean(df[['label']], min_periods=1, window=600).as_matrix()
    pos_rate2 = pos_rate2[:, 0]
    lenth=len(pos_rate)
    x=np.arange(lenth)
    f=plt.figure(figsize = (60,30),dpi = 80)
    ax=plt.subplot(331)
    ax.set_title(column)
    plt.plot(x, pos_rate,'r')
    plt.plot(x, pos_rate1, 'b')
    plt.grid()
    plt.subplot(332)
    column_value=df[column].as_matrix()
    plt.plot(x, column_value,'b')
    plt.grid()
    plt.subplot(333)
    plt.plot(x, pos_rate2, 'b')
    plt.grid()
    plt.show()


