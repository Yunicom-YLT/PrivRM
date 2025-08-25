import numpy as np
import scipy
from numpy import linalg as LA
import matplotlib.pyplot as plt

def sw(ori_samples, l, h, eps, randomized_bins=1024, domain_bins=1024):
    ee = np.exp(eps)
    w = ((eps * ee) - ee + 1) / (2 * ee * (ee - 1 - eps)) * 2
    p = ee / (w * ee + 1)
    q = 1 / (w * ee + 1)

    samples = (ori_samples - l) / (h - l)
    randoms = np.random.uniform(0, 1, len(samples))

    noisy_samples = np.zeros_like(samples)

    # report
    index = randoms <= (q * samples)
    noisy_samples[index] = randoms[index] / q - w / 2
    index = randoms > (q * samples)
    noisy_samples[index] = (randoms[index] - q * samples[index]) / p + samples[index] - w / 2
    index = randoms > q * samples + p * w
    noisy_samples[index] = (randoms[index] - q * samples[index] - p * w) / q + samples[index] + w / 2

    # report matrix
    m = randomized_bins
    n = domain_bins
    m_cell = (1 + w) / m
    n_cell = 1 / n

    transform = np.ones((m, n)) * q * m_cell
    for i in range(n):
        left_most_v = (i * n_cell)
        right_most_v = ((i + 1) * n_cell)

        ll_bound = int(left_most_v / m_cell)
        lr_bound = int((left_most_v + w) / m_cell)
        rl_bound = int(right_most_v / m_cell)
        rr_bound = int((right_most_v + w) / m_cell)

        ll_v = left_most_v - w / 2
        rl_v = right_most_v - w / 2
        l_p = ((ll_bound + 1) * m_cell - w / 2 - ll_v) * (p - q) + q * m_cell
        r_p = ((rl_bound + 1) * m_cell - w / 2 - rl_v) * (p - q) + q * m_cell
        if rl_bound > ll_bound:
            transform[ll_bound, i] = (l_p - q * m_cell) * (
                        (ll_bound + 1) * m_cell - w / 2 - ll_v) / n_cell * 0.5 + q * m_cell
            transform[ll_bound + 1, i] = p * m_cell - (p * m_cell - r_p) * (
                        rl_v - ((ll_bound + 1) * m_cell - w / 2)) / n_cell * 0.5
        else:
            transform[ll_bound, i] = (l_p + r_p) / 2
            transform[ll_bound + 1, i] = p * m_cell

        lr_v = left_most_v + w / 2
        rr_v = right_most_v + w / 2
        r_p = (rr_v - (rr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
        l_p = (lr_v - (lr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
        if rr_bound > lr_bound:
            if rr_bound < m:
                transform[rr_bound, i] = (r_p - q * m_cell) * (
                            rr_v - (rr_bound * m_cell - w / 2)) / n_cell * 0.5 + q * m_cell

            transform[rr_bound - 1, i] = p * m_cell - (p * m_cell - l_p) * (
                        (rr_bound * m_cell - w / 2) - lr_v) / n_cell * 0.5

        else:
            transform[rr_bound, i] = (l_p + r_p) / 2
            transform[rr_bound - 1, i] = p * m_cell

        if rr_bound - 1 > ll_bound + 2:
            transform[ll_bound + 2: rr_bound - 1, i] = p * m_cell

    max_iteration = 10000
    loglikelihood_threshold = 1e-3
    ns_hist, _ = np.histogram(noisy_samples, bins=randomized_bins, range=(-w / 2, 1 + w / 2))
    # return EM(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)
    return EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)
def EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold):
    # smoothing matrix
    smoothing_factor = 2
    binomial_tmp = [scipy.special.binom(smoothing_factor, k) for k in range(smoothing_factor + 1)]
    smoothing_matrix = np.zeros((n, n))
    central_idx = int(len(binomial_tmp) / 2)
    for i in range(int(smoothing_factor / 2)):
        smoothing_matrix[i, : central_idx + i + 1] = binomial_tmp[central_idx - i:]
    for i in range(int(smoothing_factor / 2), n - int(smoothing_factor / 2)):
        smoothing_matrix[i, i - central_idx: i + central_idx + 1] = binomial_tmp
    for i in range(n - int(smoothing_factor / 2), n):
        remain = n - i - 1
        smoothing_matrix[i, i - central_idx + 1:] = binomial_tmp[: central_idx + remain]
    row_sum = np.sum(smoothing_matrix, axis=1)
    smoothing_matrix = (smoothing_matrix.T / row_sum).T

    # EMS
    theta = np.ones(n) / float(n)
    theta_old = np.zeros(n)
    r = 0
    sample_size = sum(ns_hist)
    old_logliklihood = 0

    while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration:
        theta_old = np.copy(theta)
        X_condition = np.matmul(transform, theta_old)

        TMP = transform.T / X_condition

        P = np.copy(np.matmul(TMP, ns_hist))
        P = P * theta_old

        theta = np.copy(P / sum(P))

        # Smoothing step
        theta = np.matmul(smoothing_matrix, theta)
        theta = theta / sum(theta)

        logliklihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)))
        imporve = logliklihood - old_logliklihood

        if r > 1 and abs(imporve) < loglikelihood_threshold:
            # print("stop when", imporve / old_logliklihood, loglikelihood_threshold)
            break

        old_logliklihood = logliklihood

        r += 1
    return theta

def EM(n, ns_hist, transform, max_iteration, loglikelihood_threshold):
    theta = np.ones(n) / float(n)
    theta_old = np.zeros(n)
    r = 0
    sample_size = sum(ns_hist)
    old_logliklihood = 0

    while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration:
        theta_old = np.copy(theta)
        X_condition = np.matmul(transform, theta_old)

        TMP = transform.T / X_condition

        P = np.copy(np.matmul(TMP, ns_hist))
        P = P * theta_old

        theta = np.copy(P / sum(P))

        logliklihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)))
        imporve = logliklihood - old_logliklihood

        if r > 1 and abs(imporve) < loglikelihood_threshold:
            # print("stop when", imporve, loglikelihood_threshold)
            break

        old_logliklihood = logliklihood

        r += 1
    return theta
def compare_sums(ranges,arr):
    # 检查数组长度，如果是奇数，中间的元素可以归属任意一边
    mid = len(arr) // 2
    first_half_sum = sum(arr[:mid])
    second_half_sum = sum(arr[mid:])

    if first_half_sum > second_half_sum:
        b=(ranges[0]+ranges[1])/2
        return (ranges[0],b)
    else:
        a = (ranges[0] + ranges[1]) / 2
        return (a,ranges[1])
def varldp(epsilon):
    eexp=np.exp(epsilon)
    eexp2 = np.exp(epsilon / 2)
    budget = eexp
    b = (epsilon * budget - budget + 1) / (2 * budget * (budget - 1 - epsilon))
    # high_area  S_h
    p = budget / (2 * b * budget + 1)
    # low_area  S_l
    q = 1 / (2 * b * budget + 1)

    varlm=8/(epsilon**2)
    varsr=((eexp + 1) / (eexp - 1)) ** 2
    varpm=(4*eexp2) / (3 * (eexp2 - 1) ** 2)

    if epsilon<=0.61:
        varhm=varsr
    else:
        varhm=((eexp2+3)/(3*eexp2*(eexp2-1)))+(((eexp + 1) / (eexp - 1)) ** 2)/eexp2
    varsw=(4 * (q * ((1 + 3 * b + 3 * b ** 2) / 3) + p * (( 2 * b ** 3) / 3) - (q * (b + 1/2)) ** 2))/((2 * b * (p - q)) ** 2)

    return varlm,varsr,varpm,varhm,varsw

def ara_sw_number(ori_distibution,l,h,ranges,n):
     # step1:通过SW求分布
    # 查看范围内的distribution
    x = np.arange(l+0.5 * (h - l) / 1024, l+1024.5 * (h - l) / 1024, (h - l) / 1024)
    indices = np.where((x >= ranges[0]) & (x < ranges[1]))[0]
    dis_range = ori_distibution[indices]
    number = n * np.sum(dis_range) / np.sum(ori_distibution)
    return number
def ara_sw_distribution(ori_distibution,l,h,ranges):
    #step1:通过SW求分布
    #查看范围内的distribution
    x=np.arange(l+0.5*(h-l)/1024,l+1024.5*(h-l)/1024,(h-l)/1024)
    indices = np.where((x >= ranges[0]) & (x < ranges[1]))[0]
    k=len(indices)
    dis_range=ori_distibution[indices]
    expend_k=int(1024*k)
    trans=np.zeros(expend_k)


    for i in range(k):
        trans[i*1024:(i+1)*1024]=np.repeat(dis_range[i]/1024,1024)

    distribution=np.zeros(1024)
    for i in range(1024):
        distribution[i]=np.sum(trans[i*k:(i+1)*k])

    return distribution/np.sum(distribution)

def ara_sw_mean(ori_distribution, l, h, ranges):
    # step1:通过SW求分布

    # 查看范围内的distribution
    x = np.arange(l+0.5 * (h - l) / 1024, l+1024.5 * (h - l) / 1024, (h - l) / 1024)
    indices = np.where((x >= ranges[0]) & (x < ranges[1]))[0]
    dis_range = ori_distribution[indices]
    k=x[indices]


    return np.sum(np.multiply(dis_range, k))/np.sum(dis_range)

def ara_sw_sum(ori_distibution,l,h,ranges,n):
    # 查看范围内的distribution
    x = np.arange(l+0.5 * (h - l) / 1024, l+1024.5 * (h - l) / 1024, (h - l) / 1024)
    indices = np.where((x >= ranges[0]) & (x < ranges[1]))[0]
    dis_range = ori_distibution[indices]

    return np.sum(np.multiply(dis_range/np.sum(ori_distibution), indices))*n
def ara_term2(distribution,l,h,ranges,dense,n):
    #solve expected value
    # sum1=ara_sw_sum(distribution,l,h,ranges,n)
    # sum2=ara_sw_sum(distribution,l,h,dense,n)
    # n1=ara_sw_number(distribution,l,h,ranges,n)
    # n2=ara_sw_number(distribution,l,h,dense,n)
    # exv=(sum1-sum2)/(n1-n2)
    #solve term2
    x = np.arange(0.5 * (h - l) / 1024, 1024.5 * (h - l) / 1024, (h - l) / 1024)
    indices = np.where((x> ranges[0]) & (x < dense[0]) | (x > dense[1]) & (x < ranges[1]))
    if len(x[indices])==0:
        exv=0
    else:
        exv=np.sum(x[indices])/len(x[indices])
    e=0
    for i in indices[0]:
        e=e+(n*(distribution[i]/np.sum(distribution))*(x[i]))**2
    return e

def aa_irr(data,l,h,epsilon,ranges,n,nvp):

    distribution=sw(data,l,h,epsilon)
    varlm, varsr, varpm, varhm, varsw=varldp(epsilon/2)#irrnvp
    x=np.arange(l,h,(h-l)/1024)


    if nvp=='lm':
        dense=ranges
        for i in range(10):
            e1_former=n*varlm*((ranges[1]-ranges[0])/(2**(i+1)))**2
            e2_former=ara_term2(distribution,l,h,ranges,dense,n)
            e_former=e1_former+e2_former


            subdistribution = ara_sw_distribution(distribution, l, h, dense)
            dense_later=compare_sums(dense,subdistribution)
            e1_later = n * varlm * ((ranges[1] - ranges[0]) / (2 ** (i + 2))) ** 2
            e2_later=ara_term2(distribution,l,h,ranges,dense_later,n)
            e_later=e1_later+e2_later

            if e_former<e_later or i==9:
                denselm=dense
                break
            else:
                dense=dense_later
        return denselm
    
    if nvp=='sr':
        dense=ranges
        for i in range(10):
            e1_former=n*varsr*((ranges[1]-ranges[0])/(2**(i+1)))**2
            e2_former=ara_term2(distribution,l,h,ranges,dense,n)
            e_former=e1_former+e2_former


            subdistribution = ara_sw_distribution(distribution, l, h, dense)
            dense_later=compare_sums(dense,subdistribution)
            e1_later = n * varlm * ((ranges[1] - ranges[0]) / (2 ** (i + 2))) ** 2
            e2_later=ara_term2(distribution,l,h,ranges,dense_later,n)
            e_later=e1_later+e2_later

            if e_former<e_later or i==9:
                densesr=dense
                break
            else:
                dense=dense_later
        return densesr

    if nvp=='pm':
        dense=ranges
        for i in range(10):
            e1_former=n*varpm*((ranges[1]-ranges[0])/(2**(i+1)))**2
            e2_former=ara_term2(distribution,l,h,ranges,dense,n)
            e_former=e1_former+e2_former


            subdistribution = ara_sw_distribution(distribution, l, h, dense)
            dense_later=compare_sums(dense,subdistribution)
            e1_later = n * varlm * ((ranges[1] - ranges[0]) / (2 ** (i + 2))) ** 2
            e2_later=ara_term2(distribution,l,h,ranges,dense_later,n)
            e_later=e1_later+e2_later


            if e_former<e_later or i==9:
                densepm=dense
                break
            else:
                dense=dense_later
        return densepm

    if nvp=='hm':
        dense=ranges
        for i in range(10):
            e1_former=n*varhm*((ranges[1]-ranges[0])/(2**(i+1)))**2
            e2_former=ara_term2(distribution,l,h,ranges,dense,n)
            e_former=e1_former+e2_former


            subdistribution = ara_sw_distribution(distribution, l, h, dense)
            dense_later=compare_sums(dense,subdistribution)
            e1_later = n * varlm * ((ranges[1] - ranges[0]) / (2 ** (i + 2))) ** 2
            e2_later=ara_term2(distribution,l,h,ranges,dense_later,n)
            e_later=e1_later+e2_later

            if e_former<e_later or i==9:
                densehm=dense
                break
            else:
                dense=dense_later
        return densehm
 
    if nvp=='sw_unbiased':
        dense=ranges
        for i in range(10):
            e1_former=n*varsw*((ranges[1]-ranges[0])/(2**(i+1)))**2
            e2_former=ara_term2(distribution,l,h,ranges,dense,n)
            e_former=e1_former+e2_former


            subdistribution = ara_sw_distribution(distribution, l, h, dense)
            dense_later=compare_sums(dense,subdistribution)
            e1_later = n * varlm * ((ranges[1] - ranges[0]) / (2 ** (i + 2))) ** 2
            e2_later=ara_term2(distribution,l,h,ranges,dense_later,n)
            e_later=e1_later+e2_later

            if e_former<e_later or i==9:
                densesw=dense
                break
            else:
                dense=dense_later
        return densesw

    # ratelm = (ara_sw_number(distribution, l, h, ranges, n) - ara_sw_number(distribution, l, h, denselm, n)) / n

    # ratesr = (ara_sw_number(distribution, l, h, ranges, n) - ara_sw_number(distribution, l, h, densesr, n)) / n

    # ratepm = (ara_sw_number(distribution, l, h, ranges, n) - ara_sw_number(distribution, l, h, densepm, n)) / n

    # ratehm = (ara_sw_number(distribution, l, h, ranges, n) - ara_sw_number(distribution, l, h, densehm, n)) /n

    # ratesw = (ara_sw_number(distribution, l, h, ranges, n) - ara_sw_number(distribution, l, h, densesw, n)) / n

    # x = np.arange(l + 0.5 * (h - l) / 1024, l + 1024.5 * (h - l) / 1024, (h - l) / 1024)
    # indiceslm = np.where((x > ranges[0]) & (x < denselm[0]) | (x > denselm[1]) & (x < ranges[1]))
    # if len(x[indiceslm]) == 0:
    #     exvlm = 0
    # else:
    #     exvlm = np.sum(x[indiceslm]) / len(x[indiceslm])*n*ratelm

    # indicessr = np.where((x > ranges[0]) & (x < densesr[0]) | (x > densesr[1]) & (x < ranges[1]))
    # if len(x[indicessr]) == 0:
    #     exvsr = 0
    # else:
    #     exvsr = np.sum(x[indicessr]) / len(x[indicessr])*n*ratesr

    # indicespm = np.where((x > ranges[0]) & (x < densepm[0]) | (x > densepm[1]) & (x < ranges[1]))
    # if len(x[indicespm]) == 0:
    #     exvpm = 0
    # else:
    #     exvpm = np.sum(x[indicespm]) / len(x[indicespm])*n*ratepm

    # indiceshm = np.where((x > ranges[0]) & (x < densehm[0]) | (x > densehm[1]) & (x < ranges[1]))
    # if len(x[indiceshm]) == 0:
    #     exvhm = 0
    # else:
    #     exvhm = np.sum(x[indiceshm]) / len(x[indiceshm])*n*ratehm

    # indicessw = np.where((x > ranges[0]) & (x < densesw[0]) | (x > densesw[1]) & (x < ranges[1]))
    # if len(x[indicessw]) == 0:
    #     exvsw= 0
    # else:
    #     exvsw = np.sum(x[indicessw]) / len(x[indicessw])*n*ratesw



    # return denselm, densesr, densepm, densehm, densesw, ratelm, ratesr, ratepm, ratehm, ratesw,exvlm,exvsr,exvpm,exvhm,exvsw