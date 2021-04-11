from utils.loss import MSELoss, BCELoss, CELoss, logSoftMax
from utils.module import Linear, Sigmoid, TanH, Softmax,Relu, Conv1D, MaxPool1D, Flatten
from utils.toolbox import Sequentiel, Optim, SGD, shuffle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold

#mnist = fetch_openml('mnist_784', version=1, data_home='files')
#On mélange les données
data, y = shuffle(mnist.data[:3000], mnist.target[:3000])
data = np.expand_dims(data, axis = -1)
label = np.zeros((len(y), 10))
label[range(len(y)),np.intc(y)] = 1

i = data[:100]

conv = Conv1D(3, 1, 32)
tmp = conv.forward(i)
tmp2 = conv.backward_delta(i, tmp)

conv.backward_update_gradient(i, tmp)

maxpool = MaxPool1D(2,1)
d1 = maxpool.forward(i)
out = maxpool.backward_delta(i, d1)


def f(x):
    return np.argmax(x, axis = 1)

kf = KFold(n_splits=3)
#base = kf.split(data)
base = [(range(int(len(data) * 0.9)), range(int(len(data) * 0.9), len(data)))]

for id_train, id_test in base:
    
    conv    = Conv1D(3, 1, 10, init = 'xavier')
    maxpool = MaxPool1D(2,2)
    flat    = Flatten()
    h1 = Linear((2600, 100), init = 'xavier', bias = True)
    h2 = Linear((100, 10)  , init = 'xavier', bias = True)
    
    seq = Sequentiel(m=[conv, maxpool, flat, h1, Relu(), h2], a = f)
    optim = Optim(seq, logSoftMax(), 1e-3)
    
    mean, std = SGD(data[id_train], label[id_train], optim, 100, 100)

plt.plot(mean)
plt.plot(std)
plt.legend(('mean du loss', 'std du loss'))
plt.show()