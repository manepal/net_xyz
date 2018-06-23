import network as net
import numpy as np
import activation

########### initialize_parameters ###########
l = [3, 2, 1]
parameters = net.initialize_parameters(l)
for i in range(1, len(l)):
    print("W" + str(i) + ": " + str(parameters["W" + str(i)]) + "shape: " + str(parameters["W" + str(i)].shape))
    print("b" + str(i) + ": " + str(parameters["b" + str(i)]) + "shape: " + str(parameters["b" + str(i)].shape))
##############################################

########### linear_activation_forward ###########
X = np.random.randn(3, 10)
X1 = np.zeros((3, 10))

A = net.linear_activation_forward(X1, parameters["W1"], parameters["b1"], activation.sigmoid)
print("A: " + str(A))
#################################################
