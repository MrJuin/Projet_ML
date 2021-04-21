'''
Faire des tests sur les dimensions des fonctions, rapide juste un assert pour Ãªtre sur
'''
import numpy as np
from utils.loss import MSELoss
from utils.module import Linear
from utils.toolbox import SGD, Optim, Sequentiel
import matplotlib.pyplot as plt



coef1 = 1002
b = 125
# fonction linéair que l'on apprend
def f(x1):
    return x1*coef1 + b

# données d'entrainement avec bruit
def f_bruit(x1):
    bruit = np.random.normal(0,1000,len(x1)).reshape((-1,1))
    return f(x1) + bruit

nb_data = 100
data = np.random.uniform(-10,10,nb_data).reshape((-1,1))
label = f_bruit(data)
# Input and Output size of our NN
in_size  = 1
out_size = 1

# Initialize modules with respective size
iterations = 30
batch_size = 10

h1 = Linear((in_size, out_size), bias = True, init = "uniform")

seq = Sequentiel(m = [h1], a = None)
optim = Optim(seq, MSELoss(), 1e-3)

l_mean, l_sgd = SGD(data, label, optim, batch_size, iterations)

plt.plot(l_mean)
plt.show()

plt.scatter(data, label)
plt.plot()


print("parameters:",str(h1._parameters))
print("parameters:",str(h1._bias))
print("valeurs voulues:",str([[coef1], [b]]))

def en3d():
    coef1 = 1002
    coef2 = 123
    b = 125
    # fonction linéair que l'on apprend
    def f(x1,x2):
        return x1*coef1 + coef2*x2 + b
    
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
    
    h1 = Linear((in_size, out_size), bias = True, init = "uniform")
    
    seq = Sequentiel(m = [h1], a = None)
    optim = Optim(seq, MSELoss(), 1e-3)
    
    l_mean, l_sgd = SGD(data, label, optim, batch_size, iterations)
    
    plt.plot(l_mean)
    plt.show()
    
    plt.scatter(data[:,0], data[:,1])
    plt.plot()
    
    print("parameters:",str(h1._parameters))
    print("parameters:",str(h1._bias))
    print("valeurs voulues:",str([[coef1],[coef2], [b]]))