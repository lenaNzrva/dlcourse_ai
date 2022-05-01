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
        self.Layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.ReLu1 = ReLULayer()
        self.Layer2 = FullyConnectedLayer(hidden_layer_size, n_output)

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
        params = self.params()
        W1 = params["W1"]
        B1 = params["B1"]
        W2 = params["W2"]
        B2 = params["B2"]
        
        W1.grad = np.zeros_like(W1.value)
        B1.grad = np.zeros_like(B1.value)
        W2.grad = np.zeros_like(W2.value)
        B2.grad = np.zeros_like(B2.value)

        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        # forward
        L1_result = self.Layer1.forward(X)
        ReLu1_result = self.ReLu1.forward(L1_result)
        L2_result = self.Layer2.forward(ReLu1_result)
        loss, grad = softmax_with_cross_entropy(L2_result, y)
        
        
        # backward
        L2_grad = self.Layer2.backward(grad)
        ReLu1_grad = self.ReLu1.backward(L2_grad)
        L1_grad = self.Layer1.backward(ReLu1_grad)
        
        l2_W1_loss, l2_W1_grad = l2_regularization(W1.value, self.reg)
        l2_B1_loss, l2_B1_grad = l2_regularization(B1.value, self.reg)
        l2_W2_loss, l2_W2_grad = l2_regularization(W2.value, self.reg)
        l2_B2_loss, l2_B2_grad = l2_regularization(B2.value, self.reg)
        
        l2_reg = l2_W1_loss + l2_B1_loss + l2_W2_loss + l2_B2_loss 
        loss += l2_reg     
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        W1.grad += l2_W1_grad
        B1.grad += l2_B1_grad
        W2.grad += l2_W2_grad
        B2.grad += l2_B2_grad
        

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
        
        X1 = self.Layer1.forward(X)
        X_relu = self.ReLu1.forward(X1)
        X2 = self.Layer2.forward(X_relu)
        
        pred = np.argmax(X2, axis=1)
        
        
        return pred

    def params(self):
        # TODO Implement aggregating all of the params
        result = {"W1": self.Layer1.params()["W"],
                  "B1": self.Layer1.params()["B"],
                  "W2": self.Layer2.params()["W"],
                  "B2": self.Layer2.params()["B"]}

        return result
