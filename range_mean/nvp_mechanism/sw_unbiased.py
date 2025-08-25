import numpy as np


def sw_unbiased(ori_samples, l, h, eps):
  
    ee = np.exp(eps)
    w = ((eps * ee) - ee + 1) / (2 * ee * (ee - 1 - eps)) * 2 
    p = ee / (w * ee + 1)
    q = 1 / (w * ee + 1)

    samples = (ori_samples - l) / (h - l)
    randoms = np.random.uniform(0, 1, len(samples))

    noisy_samples = np.zeros_like(samples)
    index = randoms <= (q * samples)
    noisy_samples[index] = randoms[index] / q - w / 2
    index = randoms > (q * samples)
    noisy_samples[index] = (randoms[index] - q * samples[index]) / p + samples[index] - w / 2
    index = randoms > q * samples + p * w
    noisy_samples[index] = (randoms[index] - q * samples[index] - p * w) / q + samples[index] + w / 2
    
    b=w/2
    k = q * (b + 0.5)
    c = noisy_samples - k
    final = c / (2 * b * (p - q))
    final = final * (h - l) + l

    return final


def sw_unbiased_op(ori_samples1, l, h, eps):

    ori_samples=ori_samples1[(ori_samples1>=l)&(ori_samples1<=h)]

    ee = np.exp(eps)
    w = ((eps * ee) - ee + 1) / (2 * ee * (ee - 1 - eps)) * 2 
    p = ee / (w * ee + 1)
    q = 1 / (w * ee + 1)

    samples = (ori_samples - l) / (h - l)
    randoms = np.random.uniform(0, 1, len(samples))

    noisy_samples = np.zeros_like(samples)
    index = randoms <= (q * samples)
    noisy_samples[index] = randoms[index] / q - w / 2
    index = randoms > (q * samples)
    noisy_samples[index] = (randoms[index] - q * samples[index]) / p + samples[index] - w / 2
    index = randoms > q * samples + p * w
    noisy_samples[index] = (randoms[index] - q * samples[index] - p * w) / q + samples[index] + w / 2

    data_out=np.random.uniform(-w/2,1+w/2,len(ori_samples1)-len(ori_samples))
    noisy_samples1=np.concatenate([data_out,noisy_samples])

 
    b=w/2
    k = q * (b + 0.5)
    c = noisy_samples1 - k
    final = c / (2 * b * (p - q))
    final = final * (h - l) + l

    return final


# data=np.random.uniform(0,1000,100000)
# print(np.sum(data))
# print(np.sum(sw_unbiased(data,0,1000,1)))
