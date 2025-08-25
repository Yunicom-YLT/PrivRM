import numpy as np
import random
import sympy
from sympy import *
from rr import krr,krr_op
from nvp_mechanism.lm import lm,lm_op
from nvp_mechanism.sr import sr,sr_op
from nvp_mechanism.pm import pm,pm_op
from nvp_mechanism.hm import hm,hm_op
from nvp_mechanism.sw_unbiased import sw_unbiased,sw_unbiased_op
from nvp_mechanism.sw import sw
from aa.prirm_i_aa import aa_irr
from aa.prirm_op_aa import aa_op


class RangeMean():

    def __init__(self,dataname,range,epsilon,method,nvp):
        user_file_name = 'data/%s.dat' % dataname
        data_user=np.loadtxt(user_file_name)
        if dataname=='kosarak':
            l=1
            h=41270
        elif dataname=='house':
            l=0
            h=2147483600.0
        elif dataname=='fare':
            l=-362
            h=623259
        self.data=2*(data_user-l)/(h-l)-1
        self.range=range
        self.epsilon=epsilon
        self.method=method
        self.nvp=nvp
    
    def ground_truth(self):
        data_truth=self.data.copy()
        range=self.range
        elements_in_range = data_truth[(data_truth>=range[0]) & (data_truth<=range[1])]
        return np.mean(elements_in_range)

    def estimate(self):
        if self.method=='prirm_i':
            return self.prirm_i()
        elif self.method=='prirm_i_aa':
            return self.prirm_i_aa()
        elif self.method=='prirm_*':
            return self.prirm_op()
        elif self.method=='prirm_*_aa':
            return self.prirm_op_aa()
        
    def prirm_i(self):

        #stage1:
        data_rr=np.zeros(len(self.data))
        range=self.range
        condition = (self.data>=range[0]) & (self.data<=range[1])
        data_rr[condition]=1
        frequency=krr(data_rr,2,(self.epsilon)/2)
        range_count=frequency[1]*len(self.data)

        #stage2:
        data_stage2=self.data.copy()
        condition = (self.data<range[0]) & (self.data>range[1])
        data_stage2[condition]=random.uniform(range[0],range[1])
        
        if self.nvp=='lm':
            pre_sum=np.sum(lm(data_stage2,range[0],range[1],(self.epsilon)/2))
        elif self.nvp=='sr':
            pre_sum=np.sum(sr(data_stage2,range[0],range[1],(self.epsilon)/2))
        elif self.nvp=='pm':
            pre_sum=np.sum(pm(data_stage2,range[0],range[1],(self.epsilon)/2))
        elif self.nvp=='hm':
            pre_sum=np.sum(hm(data_stage2,range[0],range[1],(self.epsilon)/2))
        elif self.nvp=='sw_unbiased':
            pre_sum=np.sum(sw_unbiased(data_stage2,range[0],range[1],(self.epsilon)/2))  

        range_sum=pre_sum-((len(self.data)-range_count)*(range[0]+range[1])/2)
        range_mean=range_sum/range_count

        range_mean=min(range_mean,self.range[1])
        range_mean=max(range_mean,self.range[0])
        return range_mean
    
    def prirm_i_aa(self):

        split_index = int(0.1 * len(self.data))
        shuffled_array=self.data.copy()
        # 分割数组

        datasw = shuffled_array[:split_index]
        datanvp = shuffled_array[split_index:]

        range=aa_irr(datasw,-1,1,self.epsilon,self.range,len(self.data),self.nvp)#dense_range
        # print(range)
        #stage1:
        data_rr=np.zeros(len(datanvp))
        condition = (datanvp>=range[0]) & (datanvp<=range[1])
        data_rr[condition]=1
        frequency=krr(data_rr,2,(self.epsilon)/2)
        range_count=frequency[1]*len(datanvp)

        #stage2:
        data_stage2=datanvp.copy()
        condition = (datanvp<range[0]) & (datanvp>range[1])
        data_stage2[condition]=random.uniform(range[0],range[1])
        
        if self.nvp=='lm':
            pre_sum=np.sum(lm(data_stage2,range[0],range[1],(self.epsilon)/2))
        elif self.nvp=='sr':
            pre_sum=np.sum(sr(data_stage2,range[0],range[1],(self.epsilon)/2))
        elif self.nvp=='pm':
            pre_sum=np.sum(pm(data_stage2,range[0],range[1],(self.epsilon)/2))
        elif self.nvp=='hm':
            pre_sum=np.sum(hm(data_stage2,range[0],range[1],(self.epsilon)/2))
        elif self.nvp=='sw_unbiased':
            pre_sum=np.sum(sw_unbiased(data_stage2,range[0],range[1],(self.epsilon)/2))  

        range_sum=pre_sum-((len(datanvp)-range_count)*(range[0]+range[1])/2)
        range_mean=range_sum/range_count

        range_mean=min(range_mean,self.range[1])
        range_mean=max(range_mean,self.range[0])
        return range_mean
    


    def prirm_op(self):
        epsilon_ph=float(self.solve_ph())
        epsilon_sw=float(self.solve_sw())
        epsilon_lm=float(self.solve_lm())
        epsilon_sr=self.epsilon


        #stage1:
        range=self.range
        is_in_range = np.logical_and(self.data >= range[0], self.data <= range[1])# 生成新的数组，范围内的元素为0，不在范围内的元素为1
        data_rr = np.where(is_in_range, 0, 1)

        #stage2:
        data_stage2=self.data.copy()           
        if self.nvp=='lm':
            range_count=krr_op(data_rr,epsilon_lm)
            pre_sum=np.sum(lm_op(data_stage2,range[0],range[1],epsilon_lm))
        elif self.nvp=='sr':
            range_count=krr_op(data_rr,epsilon_sr)
            pre_sum=np.sum(sr_op(data_stage2,range[0],range[1],epsilon_sr))
        elif self.nvp=='pm':
            range_count=krr_op(data_rr,epsilon_ph)
            pre_sum=np.sum(pm_op(data_stage2,range[0],range[1],epsilon_ph))
        elif self.nvp=='hm':
            range_count=krr_op(data_rr,epsilon_ph)
            pre_sum=np.sum(hm_op(data_stage2,range[0],range[1],epsilon_ph))
        elif self.nvp=='sw_unbiased':
            range_count=krr_op(data_rr,epsilon_sw)
            pre_sum=np.sum(sw_unbiased_op(data_stage2,range[0],range[1],epsilon_sw))  

        range_sum=pre_sum-((len(self.data)-range_count)*(range[0]+range[1])/2)
        range_mean=range_sum/range_count

        range_mean=min(range_mean,self.range[1])
        range_mean=max(range_mean,self.range[0])
        return range_mean


    def prirm_op_aa(self):
        epsilon_ph=float(self.solve_ph())
        epsilon_sw=float(self.solve_sw())
        epsilon_lm=float(self.solve_lm())
        epsilon_sr=self.epsilon

        split_index = int(0.1 * len(self.data))
        shuffled_array=self.data.copy()
        # 分割数组

        datasw = shuffled_array[:split_index]
        datanvp = shuffled_array[split_index:]

        range=aa_op(datasw,-1,1,self.epsilon,self.range,len(self.data),self.nvp)#dense_range

        #stage1:
        is_in_range = np.logical_and(datanvp >= range[0], datanvp <= range[1])# 生成新的数组，范围内的元素为0，不在范围内的元素为1
        data_rr = np.where(is_in_range, 0, 1)

        #stage2:
        data_stage2=datanvp.copy()           
        if self.nvp=='lm':
            range_count=krr_op(data_rr,epsilon_lm)
            pre_sum=np.sum(lm_op(data_stage2,range[0],range[1],epsilon_lm))
        elif self.nvp=='sr':
            range_count=krr_op(data_rr,epsilon_sr)
            pre_sum=np.sum(sr_op(data_stage2,range[0],range[1],epsilon_sr))
        elif self.nvp=='pm':
            range_count=krr_op(data_rr,epsilon_ph)
            pre_sum=np.sum(pm_op(data_stage2,range[0],range[1],epsilon_ph))
        elif self.nvp=='hm':
            range_count=krr_op(data_rr,epsilon_ph)
            pre_sum=np.sum(hm_op(data_stage2,range[0],range[1],epsilon_ph))
        elif self.nvp=='sw_unbiased':
            range_count=krr_op(data_rr,epsilon_sw)
            pre_sum=np.sum(sw_unbiased_op(data_stage2,range[0],range[1],epsilon_sw))  

        range_sum=pre_sum-((len(datanvp)-range_count)*(range[0]+range[1])/2)
        range_mean=range_sum/range_count

        range_mean=min(range_mean,self.range[1])
        range_mean=max(range_mean,self.range[0])
        return range_mean



    def solve_lm(self):
        d = sympy.Symbol("d")
        k=np.exp(self.epsilon)
        f=-k + 0.5 * sympy.exp(d/2) * (1 + sympy.exp(d))
        b1 = sympy.nsolve(f, d, (0.5*self.epsilon,self.epsilon),tol=1e-6)

        a = sympy.Symbol("a")
        k=np.exp(self.epsilon)
        f1=-k + 0.25 * (sympy.exp(a/2) * (1 + sympy.exp(a)))*a/(-1+sympy.exp(a/2)+1e-10)
        b2 = sympy.nsolve(f1, a, (0.5*self.epsilon,self.epsilon),tol=1e-6)

        c = sympy.Symbol("c")
        k=np.exp(self.epsilon)
        f2=-k +4*sympy.exp(3*c/2)*(-1+sympy.exp(c/2))/(1+sympy.exp(c))/(c+1e-10)
        b3 = sympy.nsolve(f2, c, (0.5*self.epsilon,self.epsilon),tol=1e-6)

        return min(b1,b2,b3)
    
    def solve_ph(self):
        d = sympy.Symbol("d")
        k=np.exp(self.epsilon)
        f=-k + 0.5 * sympy.exp(d/2) * (1 + sympy.exp(d))
        b = sympy.nsolve(f, d, (0.5*self.epsilon,self.epsilon),tol=1e-6)
        return b

    def solve_sw(self):
        d = sympy.Symbol("d")
        k=np.exp(self.epsilon)
        f=-k + (sympy.exp(2*d)-1)/(2*d+1e-100)
        b = sympy.nsolve(f, d, (0.5*self.epsilon,self.epsilon),tol=1e-6)
        return b






