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
        layer1 = ConvolutionalLayer(in_channels=input_shape[2], out_channels=conv1_channels, filter_size=3, padding=0)
        relu1 = ReLULayer()
        max_pool1 = MaxPoolingLayer(4,4)
        layer2 = ConvolutionalLayer(in_channels=conv1_channels, out_channels=conv2_channels, filter_size=3, padding=0)
        relu2 = ReLULayer()
        max_pool2 = MaxPoolingLayer(4,4)
        flattener = Flattener()
        layer3 = FullyConnectedLayer(conv2_channels, n_output_classes)
        # TODO Create necessary layers
        # raise Exception("Not implemented!")
        
        # return 0

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

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        raise Exception("Not implemented!")

    def predict(self, X):
        # You can probably copy the code from previous assignment
        raise Exception("Not implemented!")

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        raise Exception("Not implemented!")

        return result
