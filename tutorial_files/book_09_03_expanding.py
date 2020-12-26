"""
Building on book_09_02 file to expand to multiple inputs and nodes
"""

import numpy as np

# Passed in gradients from next layer
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])



##################### Doing the input partial derivative #######################
# Input partial derivative is just the weights

# 3 neurons with 4 inputs 
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]])


dinputs = np.dot(dvalues, weights)
print("Input partial: \n",dinputs)

##################### Doing the weight partial derivative #######################
# Weight partial derivative is just the input

inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2,],
                   [-1.5, 2.7, 3.3, -0.8]])

dweights = np.dot(inputs.T, dvalues)
print("Weights Partial: \n", dweights)


##################### Doing the weight partial derivative #######################
# Biases partial == 1, so we just need to sum

biases = np.array([[2,3,0.5]])

dbiases = np.sum(dvalues, axis=0, keepdims=True)


dinputs = np.dot(inputs.T, dvalues)
print("Biases Partial: \n", dbiases)


########################## Doing the ReLU functions ##############################
z = np.array([[1, 2, -3, -4],
              [2, -7, -1, 3],
              [-1, 2, 5, -1]])

dvalues = np.array([[1, 2, 3, 4],
                    [5, 5, 6, 7],
                    [9, 10, 11, 12]])


drelu = np.zeros_like(z) # create zero'd array in shape of z 
drelu[z>0] = 1 # Where z > 0, set drelu to 1

print("Activation Mask: \n", drelu)

drelu *= dvalues


print("Post Activation Values: \n", drelu)
