# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:52:28 2021

@author: Luc
"""
from utils.loss import MSELoss, BCELoss
from utils.module import Linear, Sigmoid, TanH, Softmax
from utils.toolbox import Sequentiel, Optim, SGD

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

seq = Sequentiel(m = [h1, TanH(), h2, TanH(), h3, Softmax()], a = f)
optim = Optim(seq, MSELoss(), 0.001)
label = np.array(list(map(devellope, y)))
print("score de train avant:",optim.score_predict(data[:1000], y[:1000]))
print("score de test: avant:",optim.score_predict(data[1000:], y[1000:]))

mean, std = SGD(data[:1000], label[:1000], optim, 10, 3000)
plt.plot(mean)
plt.plot(std)
plt.legend(('mean du loss', 'std du loss'))
plt.show()
print("score de train après:",optim.score_predict(data[:1000], y[:1000]))
print("score de test après:",optim.score_predict(data[1000:], y[1000:]))