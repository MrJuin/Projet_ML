# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:52:28 2021

@author: Luc
"""
from utils.loss import MSELoss 
from utils.module import Linear, Sigmoid, TanH
from utils.toolbox import Sequentiel, Optim, SGD
import utils.graphtools as tools

import numpy as np
import matplotlib.pyplot as plt

in_size = 2
out_size = 1
h2_size = 40
h1_size = 80

def f(x):
    return np.where(x.reshape(-1) <= 0.5, -1, 1)

h1 = Linear((in_size, h1_size), bias = True,  init = 'xavier')
h2 = Linear((h1_size, h2_size), bias = True,  init = 'xavier')
h3 = Linear((h2_size, out_size), bias = True, init = 'xavier')

seq = Sequentiel(m = [h1, TanH(), h2,TanH(), h3, Sigmoid()], a = f)

optim = Optim(seq, MSELoss())


data, label = tools.gen_arti(centerx=1,centery=1,sigma=0.5,nbex=1000,data_type=1,epsilon=0.02)
def plot():
    tools.plot_frontiere(data,seq.predict,step=20)
    tools.plot_data(data,labels=label)
    plt.show()

def SGD(data, label, optim, batch_size, iterations):
    b_data  = data.reshape(-1,batch_size,in_size)
    b_label = label.reshape(-1,batch_size,out_size)
    b_label = np.where(b_label == -1, 0, 1)
    
    mean = []
    std  = []
    for i in range(iterations):
        cpt = []
        for x, y in zip(b_data, b_label):
            cpt += [optim.score(x,y)[-1]]
            optim.step(x, y)
            
        mean += [np.mean(cpt)]
        std  += [np.std(cpt)]
        plot()
        
    return mean, std

mean, std = SGD(data, label, optim, 10, 4000)
plot()
plt.plot(mean)
plt.plot(std)
plt.legend(('mean du mse', 'std du mse'))
plt.show()