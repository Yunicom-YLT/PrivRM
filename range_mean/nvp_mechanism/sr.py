import numpy as np
import random


def duchi_data(inputdata,epsilon):
    e=np.exp(epsilon)
    p=(e-1)*inputdata/(2*e+2)+0.5
    if random.random()<p:
        return (e+1)/(e-1)
    else:
        return -1*(e+1)/(e-1)


def sr(data_original, min_val, max_val,epsilon):
    normalized_data = 2*(data_original - min_val) / (max_val - min_val)-1
    for i in range(len(normalized_data)):
        normalized_data[i] = duchi_data(normalized_data[i],epsilon)
    data_noisy = (normalized_data+1)/2*(max_val - min_val)+ min_val
    return data_noisy


def sr_op(data_original1, min_val, max_val,epsilon):
    
    data_original=data_original1[(data_original1>=min_val)&(data_original1<=max_val)]

    normalized_data = 2*(data_original - min_val) / (max_val - min_val)-1
    for i in range(len(normalized_data)):
        normalized_data[i] = duchi_data(normalized_data[i],epsilon)

    e=np.exp(epsilon)
    values = [(e+1)/(e-1), -1*(e+1)/(e-1)]  
    probabilities = [0.5, 0.5]  
    data_out=np.random.choice(values, size=len(data_original1)-len(normalized_data), p=probabilities)
    normalized_data1=np.concatenate([data_out,normalized_data])

    data_noisy = (normalized_data1+1)/2*(max_val - min_val)+ min_val
    return data_noisy

# data=np.random.uniform(0,1000,100000)
# print(np.sum(data))
# print(np.sum(sr(data,0,1000,1)))

