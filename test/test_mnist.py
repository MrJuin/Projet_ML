# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:52:28 2021

@author: Luc
"""
from utils.module import MSELoss, BCELoss
from utils.module import Linear, Sigmoid, TanH, Sequentiel, Optim, Softmax
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
digits = load_digits()

tmp = list(zip(digits.data, digits.target))
np.random.shuffle(tmp)
data, y = zip(*tmp)
data = np.array(data)
data /= np.sum(data, axis = 1).reshape(-1,1)
y = np.array(y)

def devellope(x):
    a = np.zeros(10)
    a[int(x)] = 1
    return a

in_size = data.shape[1]
out_size = 10
h2_size = 30
h1_size = 40

def f(x):
    return np.argmax(x, axis = 1)

h1 = Linear((in_size, h1_size))
h2 = Linear((h1_size, h2_size))
h3 = Linear((h2_size, out_size))

seq = Sequentiel(m = [h1, Sigmoid(), h2, Sigmoid(), h3, Softmax()], a = f)
optim = Optim(seq, MSELoss(), 0.01)

def SGD(data, label, optim, batch_size, iterations):
    b_data  = data.reshape(-1,batch_size,in_size)
    l = np.array(list(map(devellope, label)))
    b_label = l.reshape(-1,batch_size,out_size)
    
    
    mean = []
    std  = []
    for i in range(iterations):
        cpt = []
        for x, y in zip(b_data, b_label):
            cpt += [optim.score(x,y)[-1]]
            optim.step(x, y)
            
        mean += [np.mean(cpt)]
        std  += [np.std(cpt)]
    return mean, std

mean, std = SGD(data[:1000], y[:1000], optim, 10, 100)
plt.plot(mean)
plt.plot(std)
plt.legend(('mean du mse', 'std du mse'))
plt.show()
