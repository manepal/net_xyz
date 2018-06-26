from network_helpers import *

class DeepNeuralNetwork():
    def __init__(self, layer_dims, activation_funcs, num_iterations, learning_rate):
        # initializes hyper parameters for the network
        #
        # arguments:
        #   - layer_dims:       list of integers in which each integer denotes the number of neurons in each layer
        #                       e.g. layer_dims = [3, 2, 1] represents a 2-Layer neural network with:
        #                           . layer_dims[0] = 3 is an input layer with 3 neurons (input features)
        #                           . layer_dims[1] = 2 is a hidden layer with 2 neurons
        #                           . layer_dims[2] = 1 is the output layer with 1 neuron
        #   - activation_funcs: list of string denoting the activation functions for each layer [1 - L]
        #   - num_iterations:   number of iterations to train the network
        #   - learning_rate:         a small scalar for updating parameters of network, determines how fast the network learns
        
        self.layer_dims = layer_dims
        self.activation_funcs = activation_funcs
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.is_training_complete = False

        # initialize w, b vectors for entire network
        self.parameters = initialize_parameters(self.layer_dims)

    def train_network(self, X_train, Y_train, cost_interval = 1000):
        # trains the network by updating the parameters W, b to reduce cost of the network
        #
        # input:
        #   - X_train:      [n x m] vector of training examples where 
        #                       . n:    number of input features in each training example
        #                       . m:    number of training examples
        #   - Y_train:      [Yn x m] vector of expected result for each training examples where
        #                       . Yn:   number of output neurons in the network
        #                       . m:    number of training examples
        #   - cost_interval:interval [in number of iteratioins] between recording the cost of network
        #
        # output:
        #   - final_cost:   cost of the network after the entire network has been trained completely
        #   - costs:        list of cost after each cost_interval
        
        if X_train.shape[0] != self.layer_dims[0]:
            print("number of input features does not match the number of input layers in the nerwork!!!")
            return

        costs = []
        final_cost = 0.0

        for i in range(self.num_iterations):
            # compute the output for all training examples at once
            AL, caches = forward_propagate(X_train, self.parameters, self.activation_funcs)
            
            # calculate the cost of the network
            cost = compute_cost(AL, Y_train)
            if i % cost_interval == 0:
                costs.append(cost)

            # compute gradients of cost
            grads = backward_propagate(AL, Y_train, caches, self.activation_funcs)

            # update the parameters
            self.parameters = update_parameters(self.parameters, grads, self.learning_rate)

        # finally after training has been completed, compute the final cost
        AL, caches = forward_propagate(X_train, self.parameters, self.activation_funcs)
        final_cost = compute_cost(AL, Y_train)

        self.is_training_complete = True

        return final_cost, costs

    def predict(self, X):
        # predicts the output of the given data based on the training completed
        #
        # input:
        #   - X:    input data to make predictions on
        # output:
        #   - AL:   activation vector of final layer

        if self.is_training_complete == False:
            print("Please train the network first!!!")
            return
        
        if X.shape[0] != self.layer_dims[0]:
            print("number of input features does not match the number of input layers in the nerwork!!!")
            return

        AL, caches = forward_propagate(X, self.parameters, self.activation_funcs)

        # return the activation vector of final layer
        return AL
