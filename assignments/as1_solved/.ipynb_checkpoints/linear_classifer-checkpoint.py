import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    try:
        predictions -= np.max(predictions, axis=1,keepdims=True)
        softmax_result = np.exp(predictions) / np.sum(np.exp(predictions),axis = 1, keepdims=True)
    except:
        predictions -= np.max(predictions, keepdims=True)
        softmax_result = np.exp(predictions)/sum(np.exp(predictions))
    return softmax_result
    # raise Exception("Not implemented!")


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    gt = np.zeros(probs.shape[0])
    gt[target_index] = 1
    cross_entropy_loss_result = -1 * np.sum(np.log(probs)*gt)
    return cross_entropy_loss_result
    # raise Exception("Not implemented!")


def softmax_with_cross_entropy(pred, target_index):
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

    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    # Without batches
#     predictions = pred.copy()
#     predictions -= np.max(predictions)
#     px = 1
#     a = np.exp(predictions)
#     b = np.sum(a)
#     c = b**-1
#     d = a[target_index] * c

#     e = np.log(d)
#     cross_entropy_loss_result = -1*px*e
    
#     ##Gradient using back propagation algoritm
#     df = 1
#     de = df*-1
#     dd = de * (1/d)
#     dc = dd*a[target_index]
#     da_target = dd*c
#     db = dc*-1*(b**-2)
#     da = np.full((predictions.shape),db)
#     da[target_index] += dd*c
#     dprediction = da*a
    
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    # With batches
    batch_size = target_index.shape[0]
    num_classes = pred.shape[1]

    predictions = pred.copy()
    predictions -= np.max(predictions, axis=1,keepdims=True)

    px = 1
    a = np.exp(predictions)
    b = np.sum(a, axis=1)
    c = b**-1
    list_a_target = np.array([a[i, target_index[i][0]] for i in range(batch_size)])
    d = list_a_target * c 
    e = np.log(d)
    g = -1*px*e
    h = np.sum(g)
    cross_entropy_loss_result = h/batch_size
    
    
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

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")
    a = W**2
    b = np.sum(a)
    loss = b*reg_strength
    
    df = 1
    dc = df*reg_strength
    db = dc*1
    grad = dc*(2*W)

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    batch_size = target_index.shape[0]
    num_classes = W.shape[1]

    predictions = np.dot(X, W)
    a = np.exp(predictions)
    b = np.sum(a, axis=1)
    c = b**-1
    list_a_target = np.array([a[i, target_index[i]] for i in range(batch_size)])
    d = list_a_target * c 
    e = np.log(d)
    g = -1*e
    h = np.sum(g)
    loss = h/batch_size
    
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

    dW = X.T @ dprediction
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            # raise Exception("Not implemented!")
            
            for b in batches_indices:
                l0, g0 = linear_softmax(X[b], self.W, y[b])
                l1, g1= l2_regularization(self.W, reg)

                g = g0+g1
                loss = l0+l1

                self.W[g>0] -=learning_rate
                self.W[g<0] +=learning_rate
                
                loss_history.append(loss)

            # end
            # print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        # y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        # raise Exception("Not implemented!")
        result = X @ self.W
        y_pred = np.argmax(result, axis=1)

        return y_pred



                
                                                          

            

                
