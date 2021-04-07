# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:52:28 2021

@author: Luc
"""
from utils.loss import MSELoss, BCELoss, CELoss, logSoftMax
from utils.module import Linear, Sigmoid, TanH, Softmax
from utils.toolbox import Sequentiel, Optim, SGD, shuffle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold

mnist = fetch_openml('mnist_784', version=1)
#On mélange les données
data, y = shuffle(mnist.data[:3000], mnist.target[:3000])

in_size = data.shape[1]
h1_size = 100
h2_size = 10

def f(x):
    return x

kf = KFold(n_splits=3)
#base = kf.split(data)
base = [(range(int(len(data) * 0.9)), range(int(len(data) * 0.9), len(data)))]

for id_train, id_test in base:
    h1 = Linear((in_size, h1_size), init = 'xavier', bias = True)
    h2 = Linear((h1_size, h2_size), init = 'xavier', bias = True)
    h3 = Linear((h2_size, h1_size), init = 'xavier', bias = True)
    h4 = Linear((h1_size, in_size), init = 'xavier', bias = True)
    
    Codeur   = [h1, TanH(), h2, TanH()]
    Decodeur = [h3, TanH(), h4, Sigmoid()]
    
    seq = Sequentiel(m = Codeur + Decodeur, a = f)
    optim = Optim(seq, MSELoss(), 1e-3)
    
    mean, std = SGD(data[id_train], data[id_train], optim, 10, 300)
    
plt.plot(mean)
plt.plot(std)
plt.legend(('mean du loss', 'std du loss'))
plt.show()