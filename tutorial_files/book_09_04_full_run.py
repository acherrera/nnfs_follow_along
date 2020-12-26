"""
Building on book_09_03 file to show a full forward and backward run
"""

import numpy as np

# Passed in gradients from next layer
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])



##################### Set up for neuron #######################
# 3 inputs with 4 values each
inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2,],
                   [-1.5, 2.7, 3.3, -0.8]])

weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

biases = np.array([[2,3,0.5]])


################### Forward pass ############################

layer_outputs  = np.dot(inputs, weights) + biases
relu_outputs = np.maximum(0, layer_outputs)


################## Backward pass #########################
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0 # Shouldn't this be covered already?

# Do the partial derivatives
dinputs = np.dot(drelu, weights.T)
dweights = np.dot(inputs.T, drelu)
dbiases = np.sum(drelu, axis=0, keepdims=True)

weights += -0.001*dweights
biases += -0.001*dbiases

print("Weights: \n", weights)
print("Biases: \n", biases)
