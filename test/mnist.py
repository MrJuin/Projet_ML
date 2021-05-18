from utils.loss import MSELoss, BCELoss, CELoss, logSoftMax
from utils.module import Linear, Sigmoid, TanH, Softmax
from utils.toolbox import Sequentiel, Optim, SGD, shuffle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold

digits = load_digits()
#On mélange les données
data, y = shuffle(digits.data, digits.target)

in_size = data.shape[1]
out_size = 10
h1_size = 40
h2_size = 30


def f(x):
    return np.argmax(x, axis = 1)

label = np.zeros((len(y), 10))
label[range(len(y)),np.intc(y)] = 1

kf = KFold(n_splits=3)
#base = kf.split(data)
base = [(range(int(len(data) * 0.9)), range(int(len(data) * 0.9), len(data)))]

score = []
for id_train, id_test in base:
    h1 = Linear((in_size, h1_size),  init = 'uniform', bias = True)
    h2 = Linear((h1_size, h2_size),  init = 'uniform', bias = True)
    h3 = Linear((h2_size, out_size), init = 'uniform', bias = True)
    def BCE():
        seq = Sequentiel(m = [h1, TanH(), h2, TanH(), h3, Softmax()], a = f)
        optim = Optim(seq, BCELoss(), 1e-2)
        return seq, optim
    
    def CE():
        seq = Sequentiel(m = [h1, TanH(), h2, TanH(), h3, Softmax()], a = f)
        optim = Optim(seq, CELoss(), 1e-3)
        return seq, optim
    
    def SoftmaxCE():
        seq = Sequentiel(m = [h1, TanH(), h2, TanH(), h3], a = f)
        optim = Optim(seq, logSoftMax(), 1e-4)
        return seq, optim
    
    seq, optim = SoftmaxCE()
    mean, std = SGD(data[id_train], label[id_train], optim, 10, 500)
    score += [optim.score_predict(data[id_test], y[id_test])]
    
plt.plot(mean)
plt.plot(std)
plt.legend(('mean du loss', 'std du loss'))
plt.show()
print("score en test:",score)
print("score de test moyen",np.mean(score))