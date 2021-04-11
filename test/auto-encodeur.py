from utils.loss import MSELoss, BCELoss, CELoss, logSoftMax
from utils.module import Linear, Sigmoid, TanH, Softmax
from utils.toolbox import Sequentiel, Optim, SGD, shuffle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold

mnist = fetch_openml('mnist_784', version=1,data_home='files')
#On mélange les données
data, y = shuffle(mnist.data[:3000], mnist.target[:3000])

in_size = data.shape[1]
h1_size = 100
h2_size = 10

kf = KFold(n_splits=3)
#base = kf.split(data)
base = [(range(int(len(data) * 0.9)), range(int(len(data) * 0.9), len(data)))]
for id_train, id_test in base:
    h1 = Linear((in_size, h1_size), init = 'xavier', bias = True)
    h2 = Linear((h1_size, h2_size), init = 'xavier', bias = True)
    
    h3 = Linear((h2_size, h1_size), init = 'xavier', bias = True)
    h3._parameters = h2._parameters.T
    h4 = Linear((h1_size, in_size), init = 'xavier', bias = True)
    h4._parameters = h1._parameters.T
    
    Codeur   = [h1, TanH(), h2, TanH()]
    Decodeur = [h3, TanH(), h4, Sigmoid()]
    
    seq = Sequentiel(m = Codeur + Decodeur)
    optim = Optim(seq, BCELoss(), 1e-3)
    
    mean, std = SGD(data[id_train], data[id_train], optim, 10, 200)

plt.plot(mean)
plt.plot(std)
plt.legend(('mean du loss', 'std du loss'))
plt.show()


def plot(sens = 'h'):
    exemple = [list(y).index(i) for i in range(10)]
    
    if sens == 'v':
        fig,ax = plt.subplots(10,2, figsize = (30,30), gridspec_kw = {'wspace' : -0.9})
        for i in range(10):
            ax[i][0].imshow(data[exemple[i]].reshape(28,28))
            ax[i][1].imshow(seq.predict(data[exemple[i]].reshape(1,-1)).reshape(28,28))

    else:
        fig,ax = plt.subplots(2,10, figsize = (30,30), gridspec_kw = {'hspace' : -0.9})
        for i in range(10):
            ax[0][i].imshow(data[exemple[i]].reshape(28,28))
            ax[1][i].imshow(seq.predict(data[exemple[i]].reshape(1,-1)).reshape(28,28))
    
    plt.show()
plot()