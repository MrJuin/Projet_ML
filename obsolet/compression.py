#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 09:19:05 2021

@author: marie
"""
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from PIL import Image
from tensorflow.keras.datasets import mnist
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tensorflow.keras.datasets import mnist
import sys
import os
sys.path.append(os.path.abspath("/dsk/win/Users/marie/Mes documents/Cours - Sorbonne/ML/Projet_ML/"))
from utils.loss import MSELoss, BCELoss, CELoss, logSoftMax
from utils.module import Linear, Sigmoid, TanH, Softmax
from utils.toolbox import Sequentiel, Optim, SGD, shuffle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


digits = load_digits()
data = digits.data
X= list(np.array(data))
np.random.shuffle(X)
X /= np.max(X)

in_size = X.shape[1]
step = -7

''' Courbe de performance 
means = []
stds = []
for i in np.arange(in_size,0,step):
    h1 = Linear((in_size, i), init = 'xavier', bias = True)
    h2 = Linear((i, in_size), init = 'xavier', bias = True)
    h2._parameters = h1._parameters.T
    Codeur   = [h1, TanH()]
    Decodeur = [h2, Sigmoid()]
    seq = Sequentiel(m = Codeur + Decodeur)
    optim = Optim(seq, BCELoss(), 1e-3)
    mean, std = SGD(X, X, optim, 10, 10)
    means.append(mean[-1])
    stds.append(std[-1])

plt.figure()
plt.plot(1-np.arange(in_size,0,step)/in_size, means)
plt.plot(1-np.arange(in_size,0,step)/in_size, stds)
plt.legend(('mean du loss', 'std du loss'))
plt.show()'''


'''Visualisation'''
def autoencoder(in_size, out_size, partage = True):
    h1 = Linear((in_size, out_size), init = 'xavier', bias = True)
    h2 = Linear((out_size, in_size), init = 'xavier', bias = True)
    if partage:
        h2._parameters = h1._parameters.T
    Codeur   = [h1, TanH()]
    Decodeur = [h2, Sigmoid()]
    seq = Sequentiel(m = Codeur + Decodeur)
    return seq

def fit(seq, X_train):
    optim = Optim(seq, BCELoss(), 1e-3)
    mean, std = SGD(X_train,X_train,optim,10,100)
    return mean[-1], std[-1]

def predict_plot(seq, X_test):
    exemple = [X_test[i] for i in range(10)]
    fig,ax = plt.subplots(2,10, figsize = (30,30), gridspec_kw = {'hspace' : -0.9})
    for i in range(10):
        ax[0][i].imshow(exemple[i].reshape(8,8), cmap='gray')
        ax[1][i].imshow(seq.predict(X_test[i].reshape(1,-1)).reshape((8,8)), cmap='gray')
    plt.show()
    

X_train = X[:int(0.8*X.shape[0])]
X_test = X[int(0.2*X.shape[0]):]

# TC = 21%
seq = autoencoder(64, 50, False) 
mean, std = fit(seq, X_train)
predict_plot(seq, X_test)
print(mean, std)

# Partage des poids
seq = autoencoder(64, 50, True) 
mean, std = fit(seq, X_train)
predict_plot(seq, X_test)
print(mean, std)
'''
# TC = 68%
seq = autoencoder(64, 20) 
mean, std = fit(seq, X_train)
predict_plot(seq, X_test)
print(mean, std)

# TC = 92%
seq = autoencoder(64, 5) 
mean, std = fit(seq, X_train)
predict_plot(seq, X_test)
print(mean, std)'''
