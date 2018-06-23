import numpy as np
import activation

def initialize_parameters(layer_dims):
    # initislizes weights(w) and biases(b) for the network
    # input:
    #   - layer_dims: list of integers in which each integer denotes the number of neurons in each layer
    #     e.g. layer_dims = [3, 2, 1] represents a 2-Layer neural network with:
    #           . layer_dims[0] = 3 is an input layer with 3 neurons (input features)
    #           . layer_dims[1] = 2 is a hidden layer with 2 neurons
    #           . layer_dims[2] = 1 is the output layer with 1 neuron
    #
    # output:
    #   - parameters: a dictionary containing matrices of W, b for each layer

    parameters = {}
    num_layers = len(layer_dims)

    for l in range(1, num_layers):
        # we start from index 1, because the first layer is input layer and does not need weights and biases

        # dimension for weights matrix is [#neurons_in_current_layer X #neurons_in_previous_layer]
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        # dimension for bias matrix is [#neurons_in_current_layer X 1]
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def linear_activation_forward(A_prev, W, b, activation_func):
    # returns the activation vector of a single layer
    #
    # input:
    #   - A_prev:           activation vector of previous layer
    #   - W:                weigts vector for the current layer
    #   - b:                bias vector for the current layer
    #   - activation_func:  activation fnction for current layer i.e. activation.sigmoid, activation.relu, etc.
    #
    # output:
    #   - A:                activation vector for current layer
    
    Z = np.dot(W, A_prev) + b
    A = activation_func(Z)

    return A
