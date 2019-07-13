# names = ['sub/decom_average.txt','sub/decom_average1.txt','sub/esim_average.txt','sub/mvlstm_average.txt']
import os
names=os.listdir('sub')

for name in names:
    print(name)
    name='sub/'+name
    result=[]
    pos_num=0
    with open(name, "r") as f:
        for line in f:
            score=float(line)
            if score>0.5:
                pos_num+=1
            result.append(score)
    print(pos_num)
    print(pos_num/5000)


