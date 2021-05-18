#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 11:59:02 2021

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

(X_train, _), (X_test, _) = mnist.load_data()
X_train=X_train.reshape(-1, 784)
X_test=X_test.reshape(-1, 784)
X_train = X_train.astype("float64")
X_test = X_test.astype("float64")
X_train /= np.max(X_train)
X_test /= np.max(X_test)
X_train = X_train[:1000]
X_test = X_test[100:]

def autoencoder(partage=True):
    # encoder
    in_size = X_train.shape[1]
    h1_size = 500
    h2_size = 100
    h3_size = 10
    
    h1 = Linear((in_size, h1_size), init = 'xavier', bias = True)
    h2 = Linear((h1_size, h2_size), init = 'xavier', bias = True)
    h3 = Linear((h2_size, h3_size), init = 'xavier', bias = True)
    
    h4 = Linear((h3_size, h2_size), init = 'xavier', bias = True)
    if partage:
        h4._parameters = h3._parameters.T
    h5 = Linear((h2_size, h1_size), init = 'xavier', bias = True)
    if partage:
        h5._parameters = h2._parameters.T
    h6 = Linear((h1_size, in_size), init = 'xavier', bias = True)
    if partage:
        h6._parameters = h1._parameters.T
    Codeur = [h1, TanH(), h2, TanH(), h3, TanH()]
    Decodeur = [h4, TanH(), h5, TanH(), h6, Sigmoid()]
    
    
    ''''MOINS DE COUCHE -> PLUS ITER
    h1_size = 100
    h2_size = 10
    
    h1 = Linear((in_size, h1_size), init = 'xavier', bias = True)
    h2 = Linear((h1_size, h2_size), init = 'xavier', bias = True)
    
    h3 = Linear((h2_size, h1_size), init = 'xavier', bias = True)
    h3._parameters = h2._parameters.T
    h4 = Linear((h1_size, in_size), init = 'xavier', bias = True)
    h4._parameters = h1._parameters.T
    Codeur = [h1, TanH(), h2, TanH()]
    Decodeur = [h3, TanH(), h4, Sigmoid()]
    '''
    
    net = Sequentiel(m = Codeur + Decodeur)
    return net

def fit(net, X_train_noisy):
    optim = Optim(net, BCELoss(), 1e-3)
    mean, std = SGD(X_train_noisy,X_train_noisy,optim, 10,10)
    return mean[-1], std[-1]
      
def predict_plot(X_test, X_test_noisy):
    exemple = [X_test[i] for i in range(10)]
    fig,ax = plt.subplots(3,10, figsize = (30,30), gridspec_kw = {'hspace' : -0.9})
    for i in range(10):
        ax[0][i].imshow(X_test[i].reshape(28,28), cmap='gray')
        ax[1][i].imshow(X_test_noisy[i].reshape((28,28)), cmap='gray')
        ax[2][i].imshow(net.predict(X_test_noisy[i].reshape(1,-1)).reshape((28,28)), cmap='gray')
    plt.show()

''' Visualisation
# 20% de bruit
NOISE_FACTOR = 0.2
X_train_noisy = X_train + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test +NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy  = np.clip(X_test_noisy, 0., 1.)
net = autoencoder() 
mean, std = fit(net, X_train_noisy)
predict_plot(X_test, X_test_noisy)
print(mean, std)
'''

''' Courbe d'erreur en fonction du bruit 
means = []
stds = []
for i in np.arange(0, 1, 0.1):
    NOISE_FACTOR = i
    X_train_noisy = X_train + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    X_test_noisy = X_test +NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
    X_train_noisy = np.clip(X_train_noisy, 0., 1.)
    X_test_noisy  = np.clip(X_test_noisy, 0., 1.)
    
    net = autoencoder()
    mean, std = fit(net,X_train_noisy)
    means.append(mean)
    stds.append(std)

plt.plot(np.arange(0,1,0.1), means)
plt.plot(np.arange(0,1,0.1), stds)
plt.legend(('mean du loss', 'std du loss'))
plt.show()'''

''' importance du partage de poids 
# 20% de bruit
NOISE_FACTOR = 0.2
X_train_noisy = X_train + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test +NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy  = np.clip(X_test_noisy, 0., 1.)

net = autoencoder(False) 
mean, std = fit(net, X_train_noisy)
predict_plot(X_test, X_test_noisy)
print(mean, std)

net = autoencoder(True) 
mean, std = fit(net, X_train_noisy)
predict_plot(X_test, X_test_noisy)
print(mean, std)'''