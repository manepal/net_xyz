import network as net
import numpy as np
import matplotlib.pyplot as plt

# this is a simple program to test the deep neural network against logical gates data

# our network will have 2 input neurons for input features or two inputs of a logic gate
# output layer will contain only one neuron to emulate logic gate's output
layer_dims = [2, 3, 1]

# length of activaiton_funcs should be [len(layer_dims) - 1] because the first layer is input layer and does not require computations
activation_funcs = ["tanh", "sigmoid"]
num_iterations = 20000
learning_rate = 0.075

# instantiate the neural network object
dnn = net.DeepNeuralNetwork(layer_dims, activation_funcs, num_iterations, learning_rate)

X_train_orig = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])
# currently the shape of our training data is [4 x 2] i.e [m x n] so, we need to transpose it to make [n x m]
X_train = X_train_orig.T

Y_train_or_orig = np.array([[0],
                            [1],
                            [1],
                            [1]])
# transpose Y_train_or_orig to get [1 x m] shape
Y_train_or = Y_train_or_orig.T

#print(str(Y_train_or.shape))

# make sure that the training X and Y have same nummber of training examples
assert(X_train.shape[1] == Y_train_or.shape[1])

final_cost, costs = dnn.train_network(X_train, Y_train_or, cost_interval=100)
print("Final cost: " + str(final_cost))
print("costs: " + str(costs))

plt.plot(costs)
plt.show()

X = np.array([[1],
              [0]])
result = np.squeeze(dnn.predict(X))
print("result: " + str(result))