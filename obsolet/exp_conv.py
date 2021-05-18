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

from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold

mnist = fetch_openml('mnist_784', version=1, data_home='files')
#%%

#On mélange les données
data, y = shuffle(mnist.data[:3000], mnist.target[:3000])
data = np.expand_dims(data, axis = -1)
label = np.zeros((len(y), 10))
label[range(len(y)),np.intc(y)] = 1
i = data[:200]

conv = Conv1D(3, 1, 32)
tmp = conv.forward(i)
tmp2 = conv.backward_delta(i, tmp)

conv.zero_grad()
conv.backward_update_gradient(i, tmp)


maxpool = MaxPool1D(2,1)
maxd1 = maxpool.forward(tmp)
out = maxpool.backward_delta(tmp, maxd1)

delta_ = tmp

#%% maxPool backward delta
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



dim = input.shape[-1]
batch = input.shape[0]
res = np.zeros(input.shape)
for i, (beg, end) in enumerate(z):
    t = np.argmax(input[:,beg:end], axis = 1)
    for j in range(input.shape[0]):
        res[j,beg + t[j],range(input.shape[-1])] += delta[j,i,:]

#%% MaxPool forward

z = zip(range(0,              input.shape[1], 1+maxpool.stride), \
        range(maxpool.k_size, input.shape[1], 1+maxpool.stride))
tmp  = np.array([np.max(input[:,beg:end], axis = 1) for beg, end in z])
res3 = np.transpose(tmp, (1,0,2))


z = zip(range(0             , input.shape[1], 1+maxpool.stride), \
        range(maxpool.k_size, input.shape[1], 1+maxpool.stride))

dim   = input.shape[-1]
batch = input.shape[0]
res4  = []
for i, (beg, end) in enumerate(z):
    t = np.argmax(input[:,beg:end], axis = 1)
    res4 += [input[np.repeat(range(batch),dim),beg + t.reshape(-1),\
         np.tile(range(dim),batch)]]
        
res4 = np.array(res4)

#%% Convolution forward
from time import time
self = conv
def conv_forw_1_t():
    t = time()
    
    X = data[:1000]
    a,_ = np.mgrid[0:X.shape[1]-self.k_size:(self.stride+1), 0:self.k_size] + np.arange(self.k_size)
    a = a.reshape(-1)
    input = X[:,a,:].reshape(X.shape[0], -1, self.k_size* self.chan_in)
    input = np.transpose(input, (1, 0, 2))
    def call(x):
        return np.dot(x,self._parameters.reshape(-1, self.chan_out))
    ret1 = np.transpose(np.array(list(map(call, input))), (1,0,2))
    return time()-t, ret1

def conv_forw_2_t():
    X = data[:1000]
    
    t = time()
    z = zip(range(0, X.shape[1], 1+ self.stride), \
            range(self.k_size, X.shape[1], 1+self.stride))
    tmp = np.array([np.dot(X[:,beg:end].reshape(-1,self.k_size*self.chan_in),\
    self._parameters.reshape(-1, self.chan_out))for beg, end in z])
    ret2 = np.transpose(tmp, (1,0,2))
    return time()-t, ret2

cpt = [[],[]]
for i in range(100):
    cpt[0] += [conv_forw_1_t()[0]]
    cpt[1] += [conv_forw_2_t()[0]]

#%% Convolution backward
def conv_back_1_t():
    input = data[:200]
    delta = delta_
    t = time()
    self._gradient = np.zeros(self._parameters.shape)
    a,_ = np.mgrid[0:input.shape[1]-self.k_size:(self.stride+1), 0:self.k_size] \
    + np.arange(self.k_size)
    a = a.reshape(-1)
    input = input[:,a,:].reshape(input.shape[0], -1, self.k_size* self.chan_in)
    input = np.transpose(input, (1, 0, 2))
    for i in range(input.shape[0]):
        self._gradient += np.dot(input[i,:].T,delta[:, i, :]).reshape(self._gradient.shape)
    
    grad1 = self._gradient.copy()
    return time()-t, grad1

def conv_back_2_t():
    input = data[:200]
    delta = delta_
    t = time()
    self._gradient = np.zeros(self._parameters.shape)
    
    z = zip(range(0, input.shape[1], 1+ self.stride), \
    range(self.k_size, input.shape[1], 1+self.stride))
    for i, (beg, end) in enumerate(z):
        self._gradient += np.dot(input[:,beg:end].reshape(input.shape[0],-1).T,\
                   delta[:, i, :]).reshape(self._gradient.shape)
            
    grad2 = self._gradient.copy()
    return time()-t, grad2
cpt = [[],[]]
for i in range(100):
    cpt[0] += [conv_back_1_t()[0]]
    cpt[1] += [conv_back_2_t()[0]]
    
    

#%%
X = input
from time import time
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

#%%
import time
ti = time.time()  
input = i
d1 = r1
g1 = np.zeros(conv._parameters.shape)

z = zip(range(0, input.shape[1], 1+ conv.stride), \
range(conv.k_size, input.shape[1], 1+conv.stride))
for i, (beg, end) in enumerate(z):
    g1 += np.dot(input[:,beg:end].reshape(input.shape[0],-1).T,\
                d1[:, i, :]).reshape(g1.shape)
print(time.time()-ti)     
        
#%% DEUXIEME
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

#%% MAXPOOL

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
