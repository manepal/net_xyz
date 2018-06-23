import numpy as np
from activation import *

# activations = {
#     "sigmoid": activation.sigmoid,
#     "tanh": activation.tanh,
#     "relu": activation.relu,
#     "leaky_relu": activation.leaky_relu
# }

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

def forward_propagate(X, parameters, activation_funcs):
    # calculates the activations for each layer
    # returns the actication for the final layer
    #
    # input:
    #   - X:            vector of training or test data, dimension: [#input_features X #training_or_test_data]
    #   - parametest:   dictionary containing the the vectors of W, b, for each layers
    #   - activation_funcs:  list of string denoting the activation functions for each layer [1 - L]
    #
    # output:
    #   - AL:           activation vector of the last layer

    # calculate the number of layers in the network
    # note that num_layers = len(layer_dims) - 1
    # because we do not count the input layer as an actual network layer
    L = len(parameters) // 2
    
    # make sure #layers == len(activations)
    assert(L == len(activation_funcs))

    A = X

    for l in range(1, L + 1):
        # iterate through 1 to L for each layer
        # retrieve parameters from the dictionary
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        # retrieve activation function from the dictionary
        activation_func = activations.get(activation_funcs[l-1])

        A = linear_activation_forward(A, W, b, activation_func)

    AL = A
    return AL

def compute_cost(AL, Y):
    # returns the discrepancy between the expected result vs actual result
    # input:
    #   - AL:   activation of the final layer or the output of the network
    #   - Y:    vector of expected output
    #
    # output:
    #   - cost: aggregate discrepancy between actual vs expected result across all training examples
    
    # m is the number of training examples

    assert(Y.shape == AL.shape)

    m = Y.shape[1]

    cost = -(1.0/m) * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))
    return cost
