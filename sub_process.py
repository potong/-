import numpy as np


result=[]
model_name='esim'
sub_file_name='sub/'+model_name+'_average_data_aug.txt'
for i in range(5):
    single_result=[]
    path_name='sub/'+model_name+str(i)+'.txt'
    f = open(path_name, encoding='utf-8')
    for line in f:
        score=float(line)
        single_result.append(score)
    result.append(single_result)
result=np.array(result)
result=result.transpose()
with open(sub_file_name, "w") as f:
    for predict in result:
       if predict.mean()>0.9999999:
           score=str(0.9999999)
       elif predict.mean()<0.0000001:
           score=str(0.0000001)
       else: score=str(predict.mean())
       f.write(score)
       f.write('\n')