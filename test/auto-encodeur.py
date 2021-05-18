from utils.loss import MSELoss, BCELoss, CELoss, logSoftMax
from utils.module import Linear, Sigmoid, TanH, Softmax, Relu
from utils.toolbox import Sequentiel, Optim, SGD, shuffle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold

mnist = fetch_openml('mnist_784', version=1,data_home='files')
#On mélange les données
#%%
data, y = shuffle(mnist.data[:3000], mnist.target[:3000])

in_size = data.shape[1]
h1_size = 500
h2_size = 210
h3_size = 40

kf = KFold(n_splits=3)
#base = kf.split(data)
base = [(range(int(len(data) * 0.9)), range(int(len(data) * 0.9), len(data)))]
for id_train, id_test in base:
    h1 = Linear((in_size, h1_size), init = 'xavier', bias = True)
    h2 = Linear((h1_size, h2_size), init = 'xavier', bias = True)
    h5 = Linear((h2_size, h3_size), init = 'xavier', bias = True)
    
    h6 = Linear((h3_size, h2_size), init = 'xavier', bias = True)
    h6._parameters = h5._parameters.T
    
    h3 = Linear((h2_size, h1_size), init = 'xavier', bias = True)
    h3._parameters = h2._parameters.T
    
    h4 = Linear((h1_size, in_size), init = 'xavier', bias = True)
    h4._parameters = h1._parameters.T
    
    Encodeur = [h1, TanH(), h2, TanH(), h5, TanH()]
    Decodeur = [h6, TanH(), h3, TanH(), h4, Sigmoid()]
    
    seq = Sequentiel(m = Encodeur + Decodeur)
    optim = Optim(seq, BCELoss(), 1e-4)
    
    mean, std = SGD(data[id_train], data[id_train], optim, 100, 10)

plt.plot(mean)
plt.plot(std)
plt.legend(('mean du loss', 'std du loss'))
plt.show()

#%%
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
#%% 

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

df_train = TSNE(n_components=2).fit_transform(data[id_train])
kmeans = KMeans(n_clusters= 10)
label = kmeans.fit_predict(df_train)
plt.figure()
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(df_train[label == i , 0] , df_train[label == i , 1] , label = i)
plt.legend()
plt.show()
#%% ESPACE LATENT

Encodeur2 = Sequentiel(Encodeur)
X_latant = np.array(Encodeur2.predict(data))
df_train_latent = TSNE(n_components=2).fit_transform(X_latant)
kmeans = KMeans(n_clusters= 10)

label = y #kmeans.fit_predict(df_train)
plt.figure()
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(df_train_latent[label == i , 0] , df_train_latent[label == i , 1] , label = i)
plt.legend()
plt.show()