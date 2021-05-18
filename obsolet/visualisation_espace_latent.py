#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 11:18:16 2021

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
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE



def autoencoder(in_size, out_size, partage = True):
    h1 = Linear((in_size, out_size), init = 'xavier', bias = True)
    h2 = Linear((out_size, in_size), init = 'xavier', bias = True)
    if partage:
        h2._parameters = h1._parameters.T
    Codeur   = [h1, TanH()]
    Decodeur = [h2, Sigmoid()]
    seq = Sequentiel(m = Codeur + Decodeur)
    return seq, [Codeur, Decodeur]

def fit(seq, X_train):
    optim = Optim(seq, BCELoss(), 1e-3)
    mean, std = SGD(X_train,X_train,optim,10,100)
    return mean[-1], std[-1]

'''
digits = load_digits()
X = digits.data
X /= np.max(X)
X_train = X[:int(0.8*X.shape[0])]
X_test = X[int(0.2*X.shape[0]):]

# ESPACE ORIGINAL
df_train = TSNE(n_components=2).fit_transform(X_train)
kmeans = KMeans(n_clusters= 10)
label = kmeans.fit_predict(df_train)

plt.figure()
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(df_train[label == i , 0] , df_train[label == i , 1] , label = i)
plt.legend()
plt.show()

# ESPACE LATENT
net, [encoder, decoder]  = autoencoder(X.shape[1], 20)
fit(net, X_train)

X_latant=[]
for i in range(len(X_train)):
    activationsE=[X_train[i].reshape(-1,1).T]
    for mod in encoder:
       activationsE += [mod.forward(activationsE[-1])]
    X_latant.append(activationsE[-1].reshape(-1))
X_latant = np.array(X_latant)


df_train_latent = TSNE(n_components=2).fit_transform(X_latant)
kmeans = KMeans(n_clusters= 10)
label = kmeans.fit_predict(df_train)

plt.figure()
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(df_train_latent[label == i , 0] , df_train_latent[label == i , 1] , label = i)
plt.legend()
plt.show()
'''

''' Classif '''
def f(x):
    return np.argmax(x, axis = 1)

def predict_(modules,X):
    activations=[X]
    for m in modules:
       activations += [m.forward(activations[-1])]
    return np.argmax(activations[-1],axis=1)

def one_hot(x):
    a = np.zeros(10)
    a[int(x)] = 1
    return a



digits = load_digits()
tmp = list(zip(digits.data, digits.target))
np.random.shuffle(tmp)
X, Y = zip(*tmp)
X /= np.max(X)
X_train = X[:int(0.8*X.shape[0])]
X_test = X[int(0.2*X.shape[0]):]
Y = np.array(Y)
Y= np.array(list(map(one_hot, Y)))
Y_train = Y[:int(0.8*X.shape[0])]
Y_test = Y[int(0.2*X.shape[0]):]

# contruction de l'espace latent
net, [encoder, decoder]  = autoencoder(X_train.shape[1], 20)
fit(net, X_train)
X_latant=[]
for i in range(len(X_train)):
    activationsE=[X_train[i].reshape(-1,1).T]
    for mod in encoder:
       activationsE += [mod.forward(activationsE[-1])]
    X_latant.append(activationsE[-1].reshape(-1))
X_latant = np.array(X_latant)

''' Classifier sur l'espace latent'''
# Classifier
h1 = Linear((20, 15), init = 'xavier', bias = True)
h2 = Linear((15, 10), init = 'xavier', bias = True)
m=[h1, TanH(), h2, Softmax()]
seq = Sequentiel(m = [h1, TanH(), h2, Softmax()], a=f)
    
optim = Optim(seq, BCELoss(), 1e-3)
SGD(X_latant,Y_train,optim,10,200)

''' Classifier sur l'espace original 
# Classifier
h1 = Linear((64, 30), init = 'xavier', bias = True)
h2 = Linear((30, 10), init = 'xavier', bias = True)
m=[h1, TanH(), h2, Softmax()]
seq = Sequentiel(m = [h1, TanH(), h2, Softmax()], a=f)
    
optim = Optim(seq, BCELoss(), 1e-3)
SGD(X_train,Y_train,optim,10,200)'''

'''Affichage 10 images'''
y=[]
yhat=[]
for i in range(10):
    activationsE=[X_test[i].reshape(-1,1).T]
    for mod in encoder:
       activationsE += [mod.forward(activationsE[-1])]
    rep_latante = activationsE[-1]
    
    yhat.append(seq.predict(rep_latante)[0])
    y.append(np.argmax(Y_test[i]))

fig,ax = plt.subplots(2,10, figsize = (30,30), gridspec_kw = {'hspace' : -0.9})
for i in range(10):
    ax[0][i].imshow(X_test[i].reshape(8,8), cmap='gray')
    ax[1][i].imshow(X_test[i].reshape((8,8)), cmap='gray')
plt.show()
    
print("Y : \t", y)
print("Yhat : \t",yhat)

''' Erreur sur 100 images 
yhat = list()
y = list()
for i in range(100):
    activationsE=[X_test[i].reshape(-1,1).T]
    for mod in encoder:
       activationsE += [mod.forward(activationsE[-1])]
    rep_latante = activationsE[-1]
    
    yhat.append(predict_(m, rep_latante)[0]) #X_test[i].reshape(-1,1).T pour classification sur l'espace originale
    y.append(np.argmax(Y_test[i]))

print(np.sum([np.where(y[i] != yhat[i],1,0) for i in range(len(yhat))])/len(yhat))'''