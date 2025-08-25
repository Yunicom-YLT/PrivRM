import numpy as np
import random



def krr(data,k,epsilon):# 0,1,2,3，，，，，

    ee=np.exp(epsilon)
    p=ee/(k-1+ee)
    q=1/(k-1+ee)
    unique_elements = np.arange(0,k)
    for i in range(len(data)):
        probability=random.random()
        if probability>p:
            data[i]= np.random.choice(unique_elements[unique_elements != data[i]])
    total_elements = len(data)
    frequencies = []
    for i in range(k):
        frequency = list(data).count(i) / total_elements
        frequencies.append(frequency)
    frequencies=(frequencies-q)/(p-q)
    frequencies[frequencies < 0] = 0
    return frequencies

def krr_op(data,epsilon):

    e=np.exp(epsilon)
    p=e/(e+1)
    data1=np.random.choice([0,1],size=np.count_nonzero(data== 0))
    data2 = np.random.choice([0, 1], size=np.count_nonzero(data == 1), p=[p, 1-p])
    data=np.concatenate([data1,data2])
    count1=np.count_nonzero(data == 1)
    true_count=(2*count1+((2*p-2)*len(data)))/(2*p-1)
    if true_count>len(data):
        true_count=len(data)

    return true_count

