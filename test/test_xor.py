# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:52:28 2021

@author: Luc
"""
from utils.loss import MSELoss
from utils.module import  Linear, TanH
import utils.graphtools as tools
from utils.toolbox import Sequentiel, Optim, SGD
import numpy as np
import matplotlib.pyplot as plt

batch_size = 10
out_size = 1
in_size = 2
h1_size = 2

mse = MSELoss()
h1 = Linear((in_size, h1_size))
tanh = TanH()
h2 = Linear((h1_size, out_size))


def f(x):
    x = h1.forward(x)
    x = tanh.forward(x)
    x = h2.forward(x)
    return np.sign(x.reshape(-1))

data, label = tools.gen_arti(centerx=1,centery=1,sigma=0.1,nbex=1000,data_type=1,epsilon=0.02)

label = label.reshape(-1, 1)

seq = Sequentiel(m = [h1, TanH(), h2], a = f)
optim = Optim(seq, MSELoss(), 0.01)

tools.plot_frontiere(data,f,step=20)
tools.plot_data(data,labels=label.reshape(-1))
plt.show()

mean, std = SGD(data, label, optim, 10, 200)

tools.plot_frontiere(data,f,step=20)
tools.plot_data(data,labels=label.reshape(-1))
plt.show()
