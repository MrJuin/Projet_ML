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
        return 2*(y-yhat)
    

class Linear(Module):
    def forward(self, X):
        """
        input  : batch * input
        output : batch * output
        """
        assert len(X.shape) == 2
        return np.dot(X, self._parameters)
    
    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        if self._gradient == None:
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
    