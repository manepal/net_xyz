import numpy as np

def cost_default(AL, Y):
    # return cost of a network
    #
    # input:
    #   - AL:   activation vector of last layer
    #   - Y:    expected output
    #
    # output:
    #   - cost: cost of network

    m = Y.shape[1]
    cost = -(1.0/m) * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))

    return cost

def cost_default_prime(AL, Y):
    # returns the derivative of default cost function
    #
    # input:
    #   - AL:   activation vector of last layer
    #   - Y:    expected output
    #
    # output:
    #   - dAL:  partial derivative of cost w.r.t. AL

    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))

    return dAL