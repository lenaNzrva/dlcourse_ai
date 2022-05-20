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
    # TODO copy from the previous assignment
    raise Exception("Not implemented!")
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
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO copy from the previous assignment
        raise Exception("Not implemented!")
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
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO copy from the previous assignment
        
        raise Exception("Not implemented!")        
        return d_input

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
        self.X = Param(X)
        batch_size, height, width, channels = self.X.value.shape

        out_height = (height - self.filter_size + 1) + self.padding*2
        out_width = (width - self.filter_size + 1) + self.padding*2
        
        W_flatten = self.W.value.reshape(self.filter_size*self.filter_size*self.in_channels, self.out_channels)
        result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        
        color = np.full(channels, 0)
        self.X_padding = np.full((batch_size,height+self.padding*2, width+self.padding*2, channels), color, dtype=np.int8)
        self.X_padding[:,self.padding:height+self.padding, self.padding:width+self.padding] = X
        
        for i in range(out_height):
            for j in range(out_width):
                X_flatten = self.X_padding[:,j:self.filter_size+j,i:self.filter_size+i,:].reshape(batch_size, self.filter_size*self.filter_size*self.in_channels)
                result[:,i,j,:] = (X_flatten@W_flatten) + self.B.value
                
        return result


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients


        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        
        batch_size, height, width, channels = self.X.value.shape
        _, out_height, out_width, out_channels = d_out.shape
        
        W_flatten = self.W.value.reshape(self.filter_size*self.filter_size*self.in_channels, self.out_channels)
        self.B.grad = np.sum(d_out, axis=(0,1,2))
        for i in range(out_height):
            for j in range(out_width):
                X_flatten = self.X_padding[:,j:self.filter_size+j,i:self.filter_size+i,:].reshape(batch_size, self.filter_size*self.filter_size*self.in_channels)
                
                # grad = d_out[:, j, i, np.newaxis, np.newaxis, np.newaxis, :]
                da_dw_flatten = X_flatten.T @ d_out.reshape((batch_size, out_height*out_width*out_channels))
                self.W.grad  += da_dw_flatten.reshape(self.filter_size, self.filter_size, channels, out_channels)
                grad = d_out[:,j:self.filter_size+j,i:self.filter_size+i,:]

                self.X.grad[:,j:self.filter_size+j,i:self.filter_size+i,:] += (grad.reshape(batch_size, out_channels) @ W_flatten.T).reshape(batch_size, height, width, channels)
   
                # grad.reshape(batch_size, out_channels) @ W_flatten.T
                # self.X.grad = 
                # print(grad.shape)
                # print(self.W.grad.shape)
                
                
                # da_dx_flatten = db_da.reshape((db_da.shape[0], db_da.shape[1]*db_da.shape[2]*db_da.shape[3]))@self.W_flatten.T
                
                

        return self.X.grad


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
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        raise Exception("Not implemented!")

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}
