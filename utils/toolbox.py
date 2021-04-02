# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:47:01 2021

@author: Luc
"""
import numpy as np

class Sequentiel:
    def __init__(self, m = None, a = None):
        self.modules = m
        self.activation = a #utiliser pour prédire

    def forward(self, x):
        l = [x]
        for m in self.modules:
            l += [m.forward(l[-1])]
        l.reverse()
        return l
        
    def backward(self, l, d0):
        """
        Prend une liste de sortie du réseau(venant de forward)
        et un delta d'une fonction de cout
        """
        d = [d0]
        for i, m in enumerate(np.flip(self.modules)):
            m.backward_update_gradient(l[i+1], d[-1])
            d += [m.backward_delta(l[i+1] , d[-1])]
    
    def apply_gradients(self, eps = 1e-2):
        for m in self.modules:
            m.update_parameters(gradient_step=eps)
            m.zero_grad()
                
    def predict(self, x):
        return self.activation(self.forward(x)[0])


class Optim:
    def __init__(self, net, loss, eps = 1e-2):
        self.net  = net
        self.loss = loss
        self.eps  = eps
        
    def step(self,b_x, b_y):
        l = self.net.forward(b_x)
        d = self.loss.backward(b_y, l[0])
        self.net.backward(l, d)
        self.net.apply_gradients(self.eps)
        
    def score(self, x, y):
        l = self.net.forward(x)
        return self.loss.forward(y, l[0])
    
    def score_predict(self, x, y):
        yhat = self.net.predict(x)
        return np.sum(yhat == y)/len(yhat)
            
def SGD(data, label, optim, batch_size, iterations):
    """
    input :
        data  : données d'entrées en taille (-1, x), x dimension des données
        label : données de sortie en taille (-1, y), y dimension des labels
        optim : un optimiseur de module
        itérations : nombre d'itérations à effectuer
        batch_size : nombre de batch à faire
        
    output :
        Un tuple contenant deux listes : 
            mean : moyenne des scores de l'optimiseur à chaque itération
            std  : écart-types des scores de l'optimiseur à chaque itération'
    """
    deb = data.shape[0]%batch_size
    b_data  = data[deb:].reshape(-1,batch_size,data.shape[-1])
    b_label = label[deb:].reshape(-1,batch_size, label.shape[-1])
    
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