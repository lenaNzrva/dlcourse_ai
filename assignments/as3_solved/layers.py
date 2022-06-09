import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    raise Exception("Not implemented!")

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    
    pred = predictions.copy()
    pred -= np.max(pred, axis=1,keepdims=True)
    
    #Softmax_with_cross_entropy
    batch_size = target_index.shape[0]
    num_classes = pred.shape[1]

    px = 1
    a = np.exp(pred)
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
    # TODO copy from the previous assignment
    # raise Exception("Not implemented!")
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X.copy()
        result = X.copy()
        result[X<0] = 0

        return result
        # raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO copy from the previous assignment
        # raise Exception("Not implemented!")
        # return d_result
        
        ones = np.ones(self.X.shape)
        ones[self.X<0] = 0
        d_result = d_out*ones
        
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = Param(X)
        result = (self.X.value @ self.W.value) + self.B.value
        return result
        # raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO copy from the previous assignment
        self.B.grad = np.sum(d_out, axis=0, keepdims=True)
        self.W.grad = self.X.value.T@d_out
        self.X.grad = d_out@self.W.value.T
        
        return self.X.grad
        # raise Exception("Not implemented!")        
        # return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))
        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = (height - self.filter_size + 1) + self.padding*2
        out_width = (width - self.filter_size + 1) + self.padding*2
        
        W_flatten = self.W.value.reshape(self.filter_size*self.filter_size*self.in_channels, self.out_channels)
        
        result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        # padding
        X_padding = np.zeros((batch_size,height+self.padding*2, width+self.padding*2, channels))
        X_padding[:,self.padding:height+self.padding, self.padding:width+self.padding] = X
        
        for i in range(out_height):
            for j in range(out_width):
                X_flatten = X_padding[:,j:self.filter_size+j,i:self.filter_size+i,:].reshape(batch_size, self.filter_size*self.filter_size*self.in_channels)
                result[:,i,j,:] = (X_flatten@W_flatten) + self.B.value
                
        self.X_cash = (X, X_padding)
        return result


    def backward(self, d_out):
        
        X,X_padding = self.X_cash
        
        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape
        
        W_flatten = self.W.value.reshape(self.filter_size*self.filter_size*channels, out_channels)
        X_grad = np.zeros_like(X_padding)
        
        self.B.grad = np.sum(d_out, axis=(0,1,2))
        for i in range(out_height):
            for j in range(out_width):
                grad = d_out[:,i,j,:]
                X_flatten = X_padding[:,j:self.filter_size+j,i:self.filter_size+i,:].reshape(batch_size,
                                                                                            self.filter_size*self.filter_size*self.in_channels)
                W_grad_flatten = X_flatten.T @ grad
                self.W.grad += W_grad_flatten.reshape(self.filter_size, self.filter_size, channels, out_channels)
                
                
                X_grad[:,j:self.filter_size+j,i:self.filter_size+i,:] += (grad @ W_flatten.T).reshape(batch_size, self.filter_size,
                                                                                                      self.filter_size, self.in_channels)

        return X_grad[:,self.padding:height+self.padding, self.padding:width+self.padding]


    def params(self):
        return { 'W': self.W, 'B': self.B }



class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        self.X = X
        batch_size, height, width, channels = X.shape
        
        self.height_out = int((height - self.pool_size)/self.stride + 1)
        self.width_out = int((width - self.pool_size)/self.stride + 1)
        self.X_pool = np.zeros((batch_size, self.height_out, self.width_out, channels))

        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        for y in range(self.height_out):
            for x in range(self.width_out):
                self.X_pool[:, y, x, :] = np.max(X[:,(y*self.pool_size):(y*self.pool_size)+self.stride,
                                              (x*self.pool_size):(x*self.pool_size)+self.stride, :], axis=(1,2))
                
        return self.X_pool
                

    def backward(self, d_out):
        grad_X = np.zeros_like(self.X)
        
        for y in range(self.height_out):
            for x in range(self.height_out):
                X_slice = self.X[:,(y*self.pool_size):(y*self.pool_size)+self.stride, 
                                 (x*self.pool_size):(x*self.pool_size)+self.stride, :]
                
                d_out_slice = d_out[:,y,x, :][:,np.newaxis, np.newaxis,:]
                
                mask = (X_slice == np.amax(X_slice, (1, 2))[:, np.newaxis, np.newaxis, :])
                grad_X[:,(y*self.pool_size):(y*self.pool_size)+self.stride, 
                       (x*self.pool_size):(x*self.pool_size)+self.stride, :] += d_out_slice*mask

        return grad_X

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        
        self.batch_size, self.height, self.width, self.channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        X_flatten = X.reshape(self.batch_size, self.height*self.width*self.channels)
        return X_flatten

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.batch_size, self.height,self.width,self.channels)

    def params(self):
        # No params!
        return {}
