import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.L1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.ReLu1 = ReLULayer()
        self.L2 = FullyConnectedLayer(hidden_layer_size, n_output)
        # raise Exception("Not implemented!")

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        # raise Exception("Not implemented!")

        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        L1_result = self.L1.forward(X)
        ReLu1_result = self.ReLu1.forward(L1_result)
        L2_result = self.L2.forward(ReLu1_result)
        
        loss0, grad0 = softmax_with_cross_entropy(L2_result, y)
        
        L2_grad = self.L2.backward(grad0)
        ReLu1_grad = self.ReLu1.backward(L2_grad)
        L1_grad = self.L1.backward(ReLu1_grad)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        loss1, grad1 = l2_regularization(L1_result, self.reg)
        loss2, grad2 = l2_regularization(L2_result, self.reg)
        
        loss = loss0+loss1+loss2 
        # raise Exception("Not implemented!")

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        raise Exception("Not implemented!")
        return pred

    def params(self):
        # TODO Implement aggregating all of the params
        result = {"L1_W":self.L1.W, "L2_W":self.L2.W}

        return result
