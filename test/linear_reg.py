'''
Faire des tests sur les dimensions des fonctions, rapide juste un assert pour Ãªtre sur
'''
import numpy as np
from utils.loss import MSELoss
from utils.module import Linear
from utils.toolbox import SGD, Optim, Sequentiel
import matplotlib.pyplot as plt



coef1 = -10
b = 0
# fonction linéair que l'on apprend
def f(x1):
    return x1*coef1 + b

# données d'entrainement avec bruit
def f_bruit(x1):
    bruit = np.random.normal(0,110,len(x1)).reshape((-1,1))
    return f(x1) + bruit

nb_data = 100
data = np.random.uniform(-10,10,nb_data).reshape((-1,1))
label = f_bruit(data)
# Input and Output size of our NN
in_size  = 1
out_size = 1

# Initialize modules with respective size
iterations = 300
batch_size = 10

h1 = Linear((in_size, out_size), bias = False, init = "uniform")

seq = Sequentiel(m = [h1], a = None)
optim = Optim(seq, MSELoss(), 1e-3)

l_mean, l_sgd = SGD(data, label, optim, batch_size, iterations)

plt.plot(l_mean)
plt.show()

plt.scatter(data, label)
x = np.array([-10, 10])

plt.plot(x,f(x), c= 'r', label = 'droite d origine')
plt.plot(x,h1.forward(x.reshape(-1,1)), c= 'g', label = 'droite trouvé')
plt.legend()
plt.show()

print("parameters:",str(h1._parameters))
print("bias:",str(h1._bias))
print("valeurs voulues:",str([[coef1], [b]]))

