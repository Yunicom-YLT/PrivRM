import numpy as np
from range_mean import RangeMean


def main(dataname,epsilon,range_size,method,nvp):

    range=(-1,-1+range_size)
    estimater=RangeMean(dataname,range,epsilon,method,nvp)
    mean_estimate=estimater.estimate()
    mean_truth=estimater.ground_truth()
    print('Ground Truth is',mean_truth)
    print('Estimated Result is',mean_estimate)
    print('The MSE is', (mean_truth-mean_estimate)**2)

if __name__ == "__main__":
    main(dataname='kosarak',epsilon=1,range_size=1,nvp='lm',method='prirm_*_aa')
