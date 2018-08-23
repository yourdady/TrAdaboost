''' 
@project TrAdaboost
@author Peng
@file softmax.py
@time 2018-06-16
'''
import numpy as np
def softmax(x):
    sum_raw = np.sum(np.exp(x),axis=-1)
    x1 = np.ones(np.shape(x))
    for i in range(np.shape(x)[0]):
        x1[i] = np.exp(x[i])/sum_raw[i]
    return x1