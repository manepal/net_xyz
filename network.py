import numpy as np
from activation import *
from cost import *

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

def linear_forward(A_prev, W, b):
    # returns the linear output of a single layer
    #
    # input:
    #   - A:        activation of previous layer
    #   - W:        weights vector for current layer
    #   - b:        bias vector for current layer
    #
    # output:
    #   - Z:        linear output of current layer
    #   - cache:    dictionary containing A_prev, W, b for current layer

    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)

    return Z, cache

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
    #   - cache:            dictionary containing linear_cache and activation_cache
    
    Z, linear_cache = linear_forward(A_prev, W, b)
    activation_cache = Z
    A = activation_func(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def forward_propagate(X, parameters, activation_funcs):
    # calculates the activations for each layer
    # returns the actication for the final layer
    #
    # input:
    #   - X:                vector of training or test data, dimension: [#input_features X #training_or_test_data]
    #   - parametest:       dictionary containing the the vectors of W, b, for each layers
    #   - activation_funcs: list of string denoting the activation functions for each layer [1 - L]
    #
    # output:
    #   - AL:               activation vector of the last layer
    #   - caches:           list of activation caches from each layyer

    # calculate the number of layers in the network
    # note that num_layers = len(layer_dims) - 1
    # because we do not count the input layer as an actual network layer
    L = len(parameters) // 2
    
    # make sure #layers == len(activations)
    assert(L == len(activation_funcs))

    caches = []
    A = X

    for l in range(1, L + 1):
        # iterate through 1 to L for each layer
        # retrieve parameters from the dictionary
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        # retrieve activation function from the dictionary
        activation_func = activations_forward.get(activation_funcs[l-1])

        A, cache = linear_activation_forward(A, W, b, activation_func)
        caches.append(cache)

    AL = A
    return AL, caches

def compute_cost(AL, Y, cost_func = cost_default):
    # returns the discrepancy between the expected result vs actual result
    # input:
    #   - AL:       activation of the final layer or the output of the network
    #   - Y:        vector of expected output
    #   - cost_func:cost function to use for calculating cost
    #
    # output:
    #   - cost: aggregate discrepancy between actual vs expected result across all training examples
    
    # m is the number of training examples

    assert(Y.shape == AL.shape)

    cost = cost_func(AL, Y)
    # convert cost into a scalar
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost

def linear_backward(dZ, cache):
    # returns the gradients of parameters for a single layer
    #
    # input:
    #   - dZ:       derivative of cost function w.r.t. linear output 'Z'
    #   - cache:    [linear cache] tuple of containing (A_prev, W, b) in the current layer
    #
    # output:
    #   - dA_prev:  partial derivative of cost w.r.t. activation of previous layer(l-1)
    #   - dW:       vector of partial derivatives of cost w.r.t. W
    #   - db:       vector of partial derivatives of cost w.r.t. W

    A_prev, W, b = cache
    # retrieve the no. of training examples
    m = A_prev.shape[1]

    dW = (1.0/m) * np.dot(dZ, A_prev.T)
    db = (1.0/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation_func_backward):
    # returns the backward linear->activation output for a single layer
    #
    # input:
    #   - dA:                       partial derivative of cost w.r.t. activation for current layer
    #   - cachhe:                   tuple of (linear_cache, activation cache)
    #   - activation_func_backward: derivative of activaiton function used in the current layer
    #
    # output:
    #   - dA_prev:                  partial derivative of cost w.r.t. activation for previous layer(l-1)
    #   - dW:                       partial derivative of cost w.r.t. W for current layer
    #   - db:                       partial derivative of cost w.r.t. b for current layer

    linear_cache, activation_cache = cache
    Z = activation_cache

    # using chain rule dL/dZ = dL/dA * dA/dZ, where L is the cost function
    dZ =  dA * activation_func_backward(Z)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def backward_propagate(AL, Y, caches, activation_funcs):
    # implements the back propagation for the entire network
    # returns gradients of cost w.r.t. W, b for each layer
    #
    # input:
    #   - AL:               activation for the final layer or the output fot the network
    #   - Y:                expected result
    #   - caches:           list containing linear_cache, activation cache for each layer
    #   - activation_funcs: list of strings denoting the activation function used for layer 1 through L
    #
    # output:
    #   - grads:            dictionary containig the partial derivatives of cost w.r.t. A, W, b for each layer

    grads = {}
    # compute number of layers
    L = len(caches)
    Y = Y.reshape(AL.shape)

    # get the partial derivative of cost w.r.t. A
    dAL = cost_default_prime(AL, Y)
    dA = dAL

    for l in reversed(range(L)):
        current_cache = caches[l]
        activation_backward = activations_backward.get(activation_funcs[l-1])
        dA_prev, dW, db = linear_activation_backward(dA, current_cache, activation_backward)
        dA = dA_prev
        grads["dA"+str(l)] = dA_prev
        grads["dW"+str(l+1)] = dW
        grads["db"+str(l+1)] = db

    return grads