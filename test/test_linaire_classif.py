# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:52:28 2021

@author: Luc
"""
from utils.loss import MSELoss
from utils.module import  Linear
import utils.graphtools as tools
import numpy as np
import matplotlib.pyplot as plt


batch_size = 1
data_size = 2

nb_h = 1

s = (batch_size, nb_h)

mse = MSELoss()

X = np.random.random((batch_size, data_size))
Y = np.random.randint(2, size=(batch_size, nb_h))

line = Linear()
line._parameters = np.array([[0.],[1.]]) #np.random.random((data_size, nb_h))

def f(X):
    h = line.forward(X)
    return np.sign(h.reshape(-1))

data, label = tools.gen_arti(centerx=1,centery=1,sigma=0.1,nbex=1000,data_type=0,epsilon=0.02)
tools.plot_frontiere(data,f,step=20)
tools.plot_data(data,labels=label)
plt.show()


iterations = 10

b_data  = data.reshape(-1,batch_size,data_size)
b_label = label.reshape(-1,batch_size,nb_h)

val = []
for i in range(iterations):
    for X, Y in zip(b_data, b_label):
        h = line.forward(X)
        s = mse.forward(h, Y)
        d2 = mse.backward(h, Y)
        d1 = line.backward_delta(X, d2)
        
        line.backward_update_gradient(X, d2)
        line.update_parameters(gradient_step=1e-3)
        line.zero_grad()
        tools.plot_frontiere(data,f,step=20)
        tools.plot_data(data,labels=label)
        plt.show()