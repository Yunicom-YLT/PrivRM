import numpy as np
from .pm import pm
from .sr import sr

def hm(data_original,min_val,max_val,epsilon):
    normalized_data = 2*(data_original - min_val) / (max_val - min_val)-1
    epsilon1=0.61
    if epsilon>epsilon1:
        perturb_p=1-np.exp(-epsilon/2)
    else:
        perturb_p=0
    mask = np.random.rand(len(normalized_data)) < perturb_p
    
    group_pm = normalized_data[mask]
    group_sr = normalized_data[~mask]

    group_pm =pm(group_pm,-1,1,epsilon)
    group_sr =sr(group_sr,-1,1,epsilon)

    recombined = np.empty_like(normalized_data)
    recombined[mask] = group_pm
    recombined[~mask] = group_sr
    data_noisy = (recombined +1)/2*(max_val - min_val)+ min_val
    return data_noisy

def hm_op(data_original1, min_val, max_val,epsilon):
    
    data_original=data_original1[(data_original1>=min_val)&(data_original1<=max_val)]
    normalized_data = 2*(data_original - min_val) / (max_val - min_val)-1

    epsilon1=0.61
    if epsilon>epsilon1:
        perturb_p=1-np.exp(-epsilon/2)
    else:
        perturb_p=0
    mask = np.random.rand(len(normalized_data)) < perturb_p
    
    group_pm = normalized_data[mask]
    group_sr = normalized_data[~mask]

    group_pm =pm(group_pm,-1,1,epsilon)
    group_sr =sr(group_sr,-1,1,epsilon)

    recombined = np.empty_like(normalized_data)
    recombined[mask] = group_pm
    recombined[~mask] = group_sr


    n_pm=int((len(data_original1)-len(data_original))*perturb_p)
    n_sr=len(data_original1)-len(data_original)-n_pm

    e=np.exp(epsilon)
    values = [(e+1)/(e-1), -1*(e+1)/(e-1)]  
    probabilities = [0.5, 0.5]  
    data_out_sr=np.random.choice(values, size=n_sr, p=probabilities)

    ee = np.exp(epsilon/2)
    s=(ee+1)/(ee-1)
    data_out_pm=np.random.uniform(-s,s,len(data_original1)-len(normalized_data))

    data_out=np.concatenate([data_out_pm,data_out_sr])
    normalized_data1=np.concatenate([data_out,recombined])

    data_noisy = (normalized_data1+1)/2*(max_val - min_val)+ min_val
    return data_noisy

# data=np.random.uniform(0,1000,100000)
# print(np.sum(data))
# print(np.sum(hm(data,0,1000,1)))