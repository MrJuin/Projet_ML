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



    
    
