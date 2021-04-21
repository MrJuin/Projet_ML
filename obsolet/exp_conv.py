# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:16:42 2021

@author: Luc
"""

from utils.loss import MSELoss, BCELoss, CELoss, logSoftMax
from utils.module import Linear, Sigmoid, TanH, Softmax,Relu, Conv1D, MaxPool1D, Flatten
from utils.toolbox import Sequentiel, Optim, SGD, shuffle
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold

i = data[:100]
tmp = conv.forward(i)
tmp2 = conv.backward_delta(i, tmp)

conv.zero_grad()
conv.backward_update_gradient(i, tmp)


maxpool = MaxPool1D(2,1)
maxd1 = maxpool.forward(tmp)
out = maxpool.backward_delta(tmp, maxd1)

raise
input = tmp
delta = maxd1

z = zip(range(0             , input.shape[1], 1+maxpool.stride), \
        range(maxpool.k_size, input.shape[1], 1+maxpool.stride))

dim   = input.shape[-1]
batch = input.shape[0]
res1  = np.zeros(input.shape)
for i, (beg, end) in enumerate(z):
    t = np.argmax(input[:,beg:end], axis = 1)
    res1[np.repeat(range(batch),dim),beg + t.reshape(-1),\
         np.tile(range(dim),batch)]\
         += delta[:,i,:].reshape(-1)


z = zip(range(0,              input.shape[1], 1+maxpool.stride), \
        range(maxpool.k_size, input.shape[1], 1+maxpool.stride))
dim = input.shape[-1]
batch = input.shape[0]
res = np.zeros(input.shape)
for i, (beg, end) in enumerate(z):
    t = np.argmax(input[:,beg:end], axis = 1)
    for j in range(input.shape[0]):
        res[j,beg + t[j],range(input.shape[-1])] += delta[j,i,:]
















"""
mnist = fetch_openml('mnist_784', version=1, data_home='files')
#On mélange les données
data, y = shuffle(mnist.data[:3000], mnist.target[:3000])
data = np.expand_dims(data, axis = -1)
label = np.zeros((len(y), 10))
label[range(len(y)),np.intc(y)] = 1
X = data[:100]



conv = Conv1D(3, 1, 32)
tmp = conv.forward(X)


tmp2 = conv.backward_delta(X, tmp)
conv.zero_grad()
conv.backward_update_gradient(X, tmp)

t5 = conv._gradient

maxpool = MaxPool1D(2,1)
maxd1 = maxpool.forward(X)
out = maxpool.backward_delta(X, maxd1)

ti = time.time()
a,_ = np.mgrid[0:X.shape[1]-conv.k_size:(conv.stride+1), 0:conv.k_size] \
    + np.arange(conv.k_size)
a = a.reshape(-1)
input = X[:,a,:].reshape(X.shape[0], -1, conv.k_size* conv.chan_in)
input = np.transpose(input, (1, 0, 2))
def call(x):
    return np.dot(x,conv._parameters.reshape(-1, conv.chan_out))
r1 = np.transpose(np.array(list(map(call, input))), (1,0,2))
print(time.time()-ti)

ti = time.time()  
z = zip(range(0, X.shape[1], 1+ conv.stride), \
        range(conv.k_size, X.shape[1], 1+conv.stride))

tmp = np.array([np.dot(X[:,beg:end].reshape(-1,conv.k_size*conv.chan_in),\
conv._parameters.reshape(-1, conv.chan_out))for beg, end in z])
r2 =  np.transpose(tmp, (1, 0, 2))
print(time.time()-ti)

###TEST GRADIENT
ti = time.time()  
input = X
d1 = r1
g1 = np.zeros(conv._parameters.shape)

z = zip(range(0, input.shape[1], 1+ conv.stride), \
range(conv.k_size, input.shape[1], 1+conv.stride))
for i, (beg, end) in enumerate(z):
    g1 += np.dot(input[:,beg:end].reshape(input.shape[0],-1).T,\
                d1[:, i, :]).reshape(g1.shape)
print(time.time()-ti)     
        
## DEUXIEME
ti = time.time()  
d1 = r1
g2 = np.zeros(conv._parameters.shape)
a,_ = np.mgrid[0:X.shape[1]-conv.k_size:(conv.stride+1), 0:conv.k_size] \
    + np.arange(conv.k_size)
a = a.reshape(-1)
input2 = X[:,a,:].reshape(X.shape[0], -1, conv.k_size* conv.chan_in)
input2 = np.transpose(input2, (1, 0, 2))



for i in range(input2.shape[0]):
    g2 += np.dot(input2[i,:].T,d1[:, i, :]).reshape(g2.shape)
print(time.time()-ti)

### MAXPOOL

input = X
delta = maxd1
print()

ti = time.time()
z = zip(range(0, input.shape[1], 1+maxpool.stride), \
range(maxpool.k_size, input.shape[1], 1+maxpool.stride))
dim = input.shape[-1]
batch = input.shape[0]

max1 = np.zeros(input.shape)
for i, (beg, end) in enumerate(z):
    t = np.argmax(input[:,beg:end], axis = 1)
    max1[np.repeat(range(batch),dim),beg + t.reshape(-1), range(dim)]\
        += delta[:,i,:].reshape(-1)
print(time.time()-ti)



"""