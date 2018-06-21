import numpy as np

def sigmoid(z):
    #          | 
    #          | .'''''   1
    #         .|          0.5
    #  .....'__|________  0
    #          |
    #          |
    #          |
    # sigmoid function
    # output ranges form 0.0 to 1.0
    # very useful for binary classification
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    # returns the derivative of sigmoid function as
    # s'(z) = s(z).(1 - s(z))
    return sigmoid(z)*(1.0 - sigmoid(z))

def tanh(z):
    #        |
    #       1|  .....
    # _______|._______
    #       .|
    #  ..... |-1
    #        |
    # hyperbolic tangent function
    # output ranges from -1.0 to 1.0
    e = np.exp(z)
    e_ = np.exp(-z)
    return (e - e_) / (e + e_)

def tanh_prime(z):
    # returns the derivative of hyperbolic tangent function as
    # tanh'(z) = 1 - tanh(z)^2
    return 1 - np.power(tanh(z), 2)

def relu(z):
    #        |    .
    #        |  .           
    # _______|.______  y = [x for x > 0
    #        |             [0 otherwise
    #        |
    #        |
    # called rectified linear unit
    # returns the maximum of (0.0, z)
    return np.maximum(0.0, z)

def relu_prime(z): 
    # returns the derivative of relu function
    # when z < 0, relu(z) = 0, a constant. so, derivative is 0
    # when z > 0, relu(z) = z, so derivative is 1
    return float(z > 0.0)   # typecast boolean value to float

def leaky_relu(z):
    #        |    .
    #        |  .           
    # _______|.______  y = [x for x > 0
    #   .  ' |             [0.001 * x otherwise
    #        |
    #        |
    # called leaky rectified linear unit
    # returns the maximum of (0.01 * z, z)
    return np.maximum(0.01 * z, z)

def leaky_relu_prime(z):
    # returns the derivative of leaky relu function
    # when z < 0, leaky_relu(z) = 0.01*z, so, derivative is 0.01
    # when z > 0, leaky_relu(z) = z, so derivative is 1
    if z > 0.0:
        return 1
    else:
        return 0.01
