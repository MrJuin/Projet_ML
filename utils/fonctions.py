# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:47:01 2021

@author: Luc
"""
import numpy as np

def SGD(data, label, optim, batch_size, iterations):
    b_data  = data.reshape(-1,batch_size,data.shape[-1])
    b_label = label.reshape(-1,batch_size, label.shape[-1])
    
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