import numpy as np

class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
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
    

class BCELoss(Loss):
    def forward(self, y, yhat, eps = 1e-10):
        """
        input  : batch*d
        output : batch
        """
        assert len(y.shape) == 2
        assert len(yhat.shape) == 2
        return - (y* np.log(yhat+eps) + (1-y)*np.log(1-yhat+eps))

    def backward(self, y, yhat, eps = 1e-10):
        return ((1-y)/(1-yhat +eps)) - (y/(yhat +eps))
    
class CELoss(Loss):
    def forward(self, y, yhat, eps = 1e-10):
        """
        input  : batch pour y et batch*d pour yhat
        output : batch
        """
        return 1-np.sum(yhat*y, axis = 1)
    
    def backward(self, y, yhat, eps = 1e-10):
        return -y
    
class logSoftMax(Loss):
    def forward(self, y, yhat, eps = 1e-10):
        return np.log(np.sum(np.exp(yhat) ,axis = 1) +eps) - np.sum(y*yhat, axis = 1)
        #exp =np.exp(yhat)
        #return np.sum(-np.log(exp/np.expand_dims(np.sum(exp ,axis = 1), axis = 1))*y, axis = 1)
    
    def backward(self, y, yhat, eps = 1e-10):
        exp =np.exp(yhat)
        return (exp/np.expand_dims(np.sum(exp ,axis = 1)+eps, axis = 1)) - y #zeros