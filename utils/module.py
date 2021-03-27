import numpy as np

class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass

class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass

class MSELoss(Loss):
    def forward(self, y, yhat):
        """
        input  : batch*d
        output : batch
        """
        assert len(y.shape) == 2
        assert len(yhat.shape) == 2
        return np.sum(np.power(y-yhat,2), axis = 1)

    def backward(self, y, yhat):
        return -2*(y-yhat)
    

class Linear(Module):    
    def __init__(self, dimensions = None):
        """
        Dimensions est un tuple (dim_in, dim_out), si les dimensions sont passées
        initialise les parameters aléatoirement
        """
        if type(dimensions) != type(None):
           self._parameters = np.random.random(dimensions)
        else:
            self._parameters = None
        self._gradient = None

    
    def forward(self, X):
        """
        input  : batch * input
        output : batch * output
        """
        assert len(X.shape) == 2
        return np.dot(X, self._parameters)
    
    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        if type(self._gradient) == type(None):
            self._gradient = np.zeros(self._parameters.shape)
        self._gradient += np.dot(input.T, delta)
    
    def backward_delta(self, input, delta):
        #Doit avoir la même dimension que l'inputS
        return np.dot(delta, self._parameters.T)
    
    def zero_grad(self):
        self._gradient = None
    
    
    
class TanH(Module):
    def forward(self, X):
        ## Calcule la passe forward
        return np.tanh(X)

    def backward_update_gradient(self, input, delta):
        ## Pas gradient pas de mise à jour
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        return delta * (1-np.power(np.tanh(input),2))
    
    def update_parameters(self, gradient_step=1e-3):
        pass
        
class Sigmoid(Module):
    def forward(self, X):
        ## Calcule la passe forward
        return 1/(1 + np.exp(-X))

    def backward_update_gradient(self, input, delta):
        ## Pas gradient pas de mise à jour
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        tmp = np.exp(-input)
        return delta * (tmp/np.power(1+tmp, 2))
    
    def update_parameters(self, gradient_step=1e-3):
        pass

class Softmax(Module):
    def forward(self, X):
        ## Calcule la passe forward
        exp = np.exp(X)
        return exp/np.expand_dims(np.sum(exp ,axis = 1), axis = 1)
    
    def backward_update_gradient(self, input, delta):
        ## Pas gradient pas de mise à jour
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        exp = np.exp(input)
        tmp = exp/np.expand_dims(np.sum(exp ,axis = 1), axis = 1)
        return delta *tmp*(1-tmp)
    
    def update_parameters(self, gradient_step=1e-3):
        pass

class BCELoss(Loss):
    def forward(self, y, yhat):
        """
        input  : batch*d
        output : batch
        """
        assert len(y.shape) == 2
        assert len(yhat.shape) == 2
        eps = 1e-10

        return - (y* np.log(yhat+eps) + (1-y)*np.log(1-yhat+eps))

    def backward(self, y, yhat):
        return ((1-y)/(1-yhat)) - (y/yhat)


class Sequentiel:
    def __init__(self, m = None, a = None):
        self.modules = m
        self.activation = a

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
    
    
