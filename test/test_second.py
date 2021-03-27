# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:52:28 2021

@author: Luc
"""
from module import MSELoss, Linear, Sigmoid, TanH
import numpy as np
import mltools as tools
import matplotlib.pyplot as plt


batch_size = 10
data_size = 2

h2_size = 1
h1_size = 5


mse = MSELoss()
h1 = Linear((data_size, h1_size))
h2 = Linear((h1_size, h2_size))
sig = Sigmoid()
tanh = TanH()

data, label = tools.gen_arti(centerx=1,centery=1,sigma=0.1,nbex=1000,data_type=0,epsilon=0.02)

def f(x):
    x = h1.forward(x)
    x = tanh.forward(x)
    x = h2.forward(x)
    x = sig.forward(x)
    return np.where(x.reshape(-1) <= 0.5, -1, 1)

def plot():
    tools.plot_frontiere(data,f,step=20)
    tools.plot_data(data,labels=label)
    plt.show()

iterations = 30

b_data  = data.reshape(-1,batch_size,data_size)
b_label = label.reshape(-1,batch_size,h2_size)
b_label = np.where(b_label == -1, 0, 1)
plot()


mean = []
std  = []
for i in range(iterations):
    cpt = []
    for x, y in zip(b_data, b_label):
        
        x1 = h1.forward(x)
        x2 = tanh.forward(x1)
        x3 = h2.forward(x2)
        x4 = sig.forward(x3)
        
        cpt += [mse.forward(y, x4)]
        
        d4 = mse.backward(x4, y)
        d3 = sig.backward_delta(x3, d4)
        d2 = h2.backward_delta(x2, d3)
        d1 = tanh.backward_delta(x1,d2)
        
        sig.backward_update_gradient(x3, d4)
        h2.backward_update_gradient(x2, d3)
        tanh.backward_update_gradient(x1, d2)
        h1.backward_update_gradient(x, d1)
        
        
        h1.update_parameters(gradient_step=1e-2)
        h2.update_parameters(gradient_step=1e-2)
        h1.zero_grad()
        h2.zero_grad()
        
    mean += [np.mean(cpt)]
    std  += [np.std(cpt)]
    plot()
    
    
plt.plot(mean)
plt.plot(std)
plt.legend(('mean du mse', 'std du mse'))
plt.show()