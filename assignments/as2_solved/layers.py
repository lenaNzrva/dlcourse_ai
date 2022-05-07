import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    
    #loss
    loss = np.sum(W**2)*reg_strength
    
    #grad 
    grad = 2*W*reg_strength

    return loss, grad


def softmax_with_cross_entropy(pred, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    
    #Normalization 
    predictions = pred.copy()
    predictions -= np.max(predictions, axis=1,keepdims=True)
    
    #Softmax_with_cross_entropy
    batch_size = target_index.shape[0]
    num_classes = pred.shape[1]

    px = 1
    a = np.exp(predictions)
    b = np.sum(a, axis=1)
    c = b**-1
    list_a_target = np.array([a[i, target_index[i]] for i in range(batch_size)])
    d = list_a_target * c 
    e = np.log(d)
    g = -1*px*e
    h = np.sum(g)
    cross_entropy_loss_result = h/batch_size
    
    #Back propagation
    df = 1
    dh = batch_size/ (batch_size**2)
    dg = 1*dh
    de = dg*-1
    dd = de * (1/d)
    dc = dd*list_a_target
    da_target = dd*c
    db = dc*-1*(b**-2)
    db = db.reshape((batch_size, 1))
    da = np.full((batch_size, num_classes),db)
    for i in range(batch_size):
        da[i, target_index[i]] += (dd*c)[i]
        
    dprediction = da*a
    
    return cross_entropy_loss_result, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        
        self.X = X.copy()
        result = X.copy()
        result[X<0] = 0

        return result

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        # raise Exception("Not implemented!")
        
        ones = np.ones(self.X.shape)
        ones[self.X<0] = 0
        d_result = d_out*ones
        
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = Param(X)
        result = (self.X.value @ self.W.value) + self.B.value
        return result

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        
        self.B.grad = np.sum(d_out, axis=0, keepdims=True)
        self.W.grad = self.X.value.T@d_out
        self.X.grad = d_out@self.W.value.T
        
        return self.X.grad

    def params(self):
        return {'W': self.W, 'B': self.B}
