import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        self.layer1 = ConvolutionalLayer(in_channels=input_shape[2], out_channels=conv1_channels, filter_size=3, padding=0)
        self.relu1 = ReLULayer()
        self.max_pool1 = MaxPoolingLayer(4,4)
        self.layer2 = ConvolutionalLayer(in_channels=conv1_channels, out_channels=conv2_channels, filter_size=3, padding=0)
        self.relu2 = ReLULayer()
        self.max_pool2 = MaxPoolingLayer(4,4)
        self.flattener = Flattener()
        self.layer3 = FullyConnectedLayer(conv2_channels, n_output_classes)


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        params = self.params()
        W1 = params["W1"]
        B1 = params["B1"]
        W2 = params["W2"]
        B2 = params["B2"]
        W3 = params["W3"]
        B3 = params["B3"]
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        
        W1.grad = np.zeros_like(W1.value)
        B1.grad = np.zeros_like(B1.value)
        W2.grad = np.zeros_like(W2.value)
        B2.grad = np.zeros_like(B2.value)
        W3.grad = np.zeros_like(W3.value)
        B3.grad = np.zeros_like(B3.value)

        l1 = self.layer1.forward(X)
        r1 = self.relu1.forward(l1)
        m_p1 = self.max_pool1.forward(r1)
        l2 = self.layer2.forward(m_p1)
        r2 = self.relu2.forward(l2)
        m_p2 = self.max_pool2.forward(r2)
        f = self.flattener.forward(m_p2)
        l3 = self.layer3.forward(f)
        
        loss, grad = softmax_with_cross_entropy(l3, y)
        
        g_l3 = self.layer3.backward(grad)
        g_f = self.flattener.backward(g_l3)
        g_m_p2 = self.max_pool2.backward(g_f)
        g_r2 = self.relu2.backward(g_m_p2)
        g_l2 = self.layer2.backward(g_r2)
        g_m_p1 = self.max_pool1.backward(g_l2)
        g_r1 = self.relu1.backward(g_m_p1)
        g_l1 = self.layer1.backward(g_r1)
        
        return loss

    def predict(self, X):
        l1 = self.layer1.forward(X)
        r1 = self.relu1.forward(l1)
        m_p1 = self.max_pool1.forward(r1)
        l2 = self.layer2.forward(m_p1)
        r2 = self.relu2.forward(l2)
        m_p2 = self.max_pool2.forward(r2)
        f = self.flattener.forward(m_p2)
        l3 = self.layer3.forward(f)
    
        pred = np.argmax(l3, 1)
        return pred
        

    def params(self):
        result = {"W1": self.layer1.params()["W"],
                 "B1": self.layer1.params()["B"],
                 "W2": self.layer2.params()["W"],
                 "B2": self.layer2.params()["B"],
                 "W3": self.layer3.params()["W"],
                 "B3": self.layer3.params()["B"]}


        return result
