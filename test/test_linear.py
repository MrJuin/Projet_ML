'''
Faire des tests sur les dimensions des fonctions, rapide juste un assert pour Ãªtre sur
'''
import numpy as np

import utils.mltools as tools
from utils.module import MSELoss, Optim, Sequentiel
from utils.module import Linear
from utils.fonctions import SGD
import matplotlib.pyplot as plt

coef1 = 56
coef2 = 24
# fonction linéair que l'on apprend
def f(x1,x2):
    return x1*coef1 + coef2*x2

# données d'entrainement avec bruit
def f_bruit(x1,x2):
    bruit = np.random.normal(0,1,len(x1)).reshape((-1,1))
    return f(x1,x2) + bruit

nb_data = 100
x1 = np.random.uniform(-10,10,nb_data).reshape((-1,1))
x2 = np.random.uniform(-10,10,nb_data).reshape((-1,1))
label = f_bruit(x1,x2)
data = np.concatenate((x1,x2),axis=1)
# Input and Output size of our NN
in_size  = 2
out_size = 1

# Initialize modules with respective size
iterations = 30
batch_size = 10
h1 = Linear((in_size, out_size))
seq = Sequentiel(m = [h1], a = None)
optim = Optim(seq, MSELoss(), 0.0001)

l_mean, l_sgd = SGD(data, label, optim, batch_size, iterations)

plt.plot(l_mean)
plt.show()

print("parameters:",str(h1._parameters))
print("valeurs voulues:",str([[coef1],[coef2]]))