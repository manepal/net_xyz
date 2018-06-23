import network as net
import numpy as np
import activation

########### common test data ###########
layer_dims = [2, 5, 4, 3, 2, 1]

activations = []
for l in range(len(layer_dims) - 2):
    activations.append("leaky_relu")
activations.append("sigmoid")

parameters = net.initialize_parameters(layer_dims)

# input data form logic gates
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
X = X.T
print(str(X) + "Shape of X: " + str(X.shape))
########################################


########### initialize_parameters ###########
def test_initialize_parameters():
    for i in range(1, len(layer_dims)):
        print("W" + str(i) + ": " + str(parameters["W" + str(i)]) + "shape: " + str(parameters["W" + str(i)].shape))
        print("b" + str(i) + ": " + str(parameters["b" + str(i)]) + "shape: " + str(parameters["b" + str(i)].shape))
##############################################


########### linear_activation_forward ###########
def test_linear_activation_forward():
    X1 = np.zeros((3, 10))

    A = net.linear_activation_forward(X1, parameters["W1"], parameters["b1"], activation.sigmoid)
    print("A: " + str(A))
#################################################


########### forward_propagate ###########
def test_forward_propagate():
    AL = net.forward_propagate(X, parameters, activations)
    print("AL: " + str(AL))
#########################################

test_forward_propagate()