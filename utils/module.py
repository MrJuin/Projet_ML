import numpy as np

class Module(object):
    
    def initialise(dimensions, type_):
        if type(dimensions) == type(None) or type(type_) == type(None):
            return None
        if type_ == "randn":
            return np.random.standard_normal(dimensions)
        if type_ == 'xavier':
            return np.random.standard_normal(dimensions)*np.sqrt(2/sum(dimensions))
        if type_ == 'xavier_tanh':
            return np.random.standard_normal(dimensions)*np.sqrt(2)*np.sqrt(2/sum(dimensions))
        if type_ == 'uniform':
            return (np.random.random(dimensions) -0.5)
        raise Exception('initialisation inconnue')            
        
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

class Linear(Module):    
    def __init__(self, dimensions = None, init = 'xavier', bias = True):
        """
        Dimensions : un tuple (dim_in, dim_out), si les dimensions sont passées
        initialise les parameters aléatoirement selon la méthode définit dans init,
        uniforme sinon.
        bias : if true, ajoute un bias au module
        
        """
        self._gradient   = None

        self._parameters = Module.initialise(dimensions, init)
        if type(self._parameters) != type(None) and bias:
            self._bias = Module.initialise((1,dimensions[1]), init)
        else :
            self._bias = None            
    
    def forward(self, X):
        """
        input  : batch * input
        output : batch * output
        """
        assert len(X.shape) == 2
        tmp = np.dot(X, self._parameters)
        if type(self._bias) != type(None):
            return tmp + self._bias
        return tmp
    
    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient
        
        if type(self._bias) != type(None):
            self._bias -=gradient_step*self._biasgrad
    
    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        try:
            self._gradient += np.dot(input.T, delta)
        except (AttributeError, TypeError): # Si _gradient n'existe pas, ou si il vaut None
            self._gradient = np.dot(input.T, delta)
        
        if type(self._bias) != type(None):
            try:
                self._biasgrad += np.sum(delta, axis = 0)
            except (AttributeError, TypeError):
                self._biasgrad = np.sum(delta, axis = 0)
                
    def backward_delta(self, input, delta):
        #Doit avoir la même dimension que l'inputS
        return np.dot(delta, self._parameters.T)
    
    def zero_grad(self):
        self._biasgrad = None
        self._gradient = None



class Conv1D(Module):    
    def __init__(self, k_size, chan_in, chan_out, stride = 0, init = 'xavier'):
        """
        Dimensions : un tuple (dim_in, dim_out), si les dimensions sont passées
        initialise les parameters aléatoirement selon la méthode définit dans init,
        uniforme sinon.
        bias : if true, ajoute un bias au module
        
        """
        self._gradient   = None
        self._parameters = Module.initialise((k_size, chan_in, chan_out), init)
        self.k_size      = k_size
        self.stride      = stride
        self.chan_in     = chan_in
        self.chan_out    = chan_out
    
    def forward(self, X):
        """
        input  : batch * input * chan_in
        output : batch * (input - k_size/stride + 1)* chan_out
        """
        assert X.shape[2] == self.chan_in
        z = zip(range(0, X.shape[1], 1+ self.stride), \
                range(self.k_size, X.shape[1], 1+self.stride))
            
        tmp = np.array([np.dot(X[:,beg:end].reshape(-1,self.k_size*self.chan_in),\
        self._parameters.reshape(-1, self.chan_out))for beg, end in z])
        return tmp.reshape(X.shape[0],-1,self.chan_out)
    
    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient
    
    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        if self._gradient is None: 
            self._gradient = np.zeros(self._parameters.shape)
            
        z = zip(range(0, input.shape[1], 1+ self.stride), \
        range(self.k_size, input.shape[1], 1+self.stride))        
        for i, (beg, end) in enumerate(z):
            self._gradient += np.dot(input[:,beg:end].reshape(input.shape[0],-1).T,\
                       delta[:, i, :]).reshape(self._gradient.shape)
        
    def backward_delta(self, input, delta):
        z = zip(range(0, input.shape[1], 1+ self.stride), \
        range(self.k_size, input.shape[1], 1+self.stride))
        res = np.zeros(input.shape)
        for i, (begin, end) in enumerate(z):
            d = np.dot(delta[:, i, :], \
                self._parameters.reshape(-1, self.chan_out).T)
            res[:,begin:end] += d.reshape(-1, self.k_size, self.chan_in)           
        return res
    
    def zero_grad(self):
        self._gradient = None

class MaxPool1D(Module):    
    def __init__(self, k_size, stride = 0):
        """
        Dimensions : un tuple (dim_in, dim_out), si les dimensions sont passées
        initialise les parameters aléatoirement selon la méthode définit dans init,
        uniforme sinon.
        bias : if true, ajoute un bias au module
        
        """
        self.k_size = k_size
        self.stride = stride
    
    def forward(self, X):
        """
        input  : batch * input * chan_in
        output : batch * (input - k_size/stride + 1)* chan_out
        """
        z = zip(range(0, X.shape[1], 1+self.stride), \
                range(self.k_size, X.shape[1], 1+self.stride))
            
        tmp = np.array([np.max(X[:,beg:end], axis = 1) for beg, end in z])
        return tmp.reshape(X.shape[0],-1,X.shape[2])
    
    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        pass
            
    def backward_delta(self, input, delta):
        #Doit avoir la même dimension que l'inputS
        z = zip(range(0, input.shape[1], 1+self.stride), \
        range(self.k_size, input.shape[1], 1+self.stride))

        res = np.zeros(input.shape)
        for i, (beg, end) in enumerate(z):
            t = np.argmax(input[:,beg:end], axis = 1)
            for j in range(input.shape[0]):
                res[j,beg + t[j],range(input.shape[-1])] += delta[j,i,:]
            
            #res[:,beg :end,:][t,range(len(t[0])), range(len(t[0]))] += delta[:,i,:]           
        return res


class Flatten(Module):
    def forward(self, X):
        return X.reshape(len(X),-1)
    
    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        
        return delta.reshape(input.shape)
    
    def update_parameters(self, gradient_step=1e-3):
        pass


class TanH(Module):
    def forward(self, X):
        ## Calcule la passe forward
        return np.tanh(X)
    
    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        return delta * (1-np.power(np.tanh(input),2))
    
    def update_parameters(self, gradient_step=1e-3):
        pass
        
class Sigmoid(Module):
    def forward(self, X):
        ## Calcule la passe forward
        return 1/(1 + np.exp(-X))

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

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        exp = np.exp(input)
        tmp = exp/np.expand_dims(np.sum(exp ,axis = 1), axis = 1)
        return delta *tmp*(1-tmp)
    
    def update_parameters(self, gradient_step=1e-3):
        pass

class Relu(Module):
    def forward(self, X):
        return np.where(X < 0, 0, X)
    
    def backward_delta(self, input, delta):
        return delta*np.where(input < 0, 0, 1)
        
    def update_parameters(self, gradient_step=1e-3):
        pass