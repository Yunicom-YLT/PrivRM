import numpy as np
import random

def disturb_data(data,ee):
    lv=(ee*data-1)/(ee-1)
    rv=(ee*data+1)/(ee-1)
    s=(ee+1)/(ee-1)

    if random.random() < ee/(ee+1):
        return random.uniform(lv, rv)
    else:
        if random.random() < (data+1)/2:
           return random.uniform(-s, lv)
        else:
           return random.uniform(rv, s)
        
def pm(data_original,min_val,max_val,epsilon):
    normalized_data = 2*(data_original - min_val) / (max_val - min_val)-1
    ee = np.exp(epsilon/2)
    for i in range(len(normalized_data)):
        normalized_data[i]=disturb_data(normalized_data[i],ee)
    data_noisy = (normalized_data+1)/2*(max_val - min_val)+ min_val
    return data_noisy

def pm_op(data_original1, min_val, max_val,epsilon):
    
    data_original=data_original1[(data_original1>=min_val)&(data_original1<=max_val)]

    normalized_data = 2*(data_original - min_val) / (max_val - min_val)-1
    ee = np.exp(epsilon/2)
    s=(ee+1)/(ee-1)
    for i in range(len(normalized_data)):
        normalized_data[i]=disturb_data(normalized_data[i],ee)


    data_out=np.random.uniform(-s,s,len(data_original1)-len(normalized_data))
    normalized_data1=np.concatenate([data_out,normalized_data])

    data_noisy = (normalized_data1+1)/2*(max_val - min_val)+ min_val
    return data_noisy


# data=np.random.uniform(0,1000,100000)
# print(np.sum(data))
# print(np.sum(pm(data,0,1000,1)))