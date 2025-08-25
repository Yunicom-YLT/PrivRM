import numpy as np

def lm(data_original, min_val, max_val,epsilon):
    normalized_data = 2*(data_original - min_val) / (max_val - min_val)-1
    for i in range(len(normalized_data)):
        normalized_data[i] += np.random.laplace(0, 2 / epsilon, 1)
    data_noisy = (normalized_data+1)/2*(max_val - min_val)+ min_val
    return data_noisy

def lm_op(data_original1, min_val, max_val,epsilon):
    
    data_original=data_original1[(data_original1>=min_val)&(data_original1<=max_val)]

    normalized_data = 2*(data_original - min_val) / (max_val - min_val)-1
    for i in range(len(normalized_data)):
        normalized_data[i] += np.random.laplace(0, 2 / epsilon, 1)

    normalized_data[normalized_data>1]=1
    normalized_data[normalized_data<-1]=-1

    data_out=np.random.uniform(-1,1,len(data_original1)-len(normalized_data))
    normalized_data1=np.concatenate([data_out,normalized_data])

    data_noisy = (normalized_data1+1)/2*(max_val - min_val)+ min_val
    return data_noisy


# data=np.random.uniform(0,1000,100000)
# print(np.sum(data))
# print(np.sum(lm(data,0,1000,1)))

