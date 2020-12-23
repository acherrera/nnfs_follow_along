"""
Using Activation Functions in the network.
* Step function - 0 or 1. Not granular enough
* Sigmoud - Gradual change from 0 to 1 output. Has a problem that makes
Rectified Linear looks better
* Rectified Linear - output is X when X > 0, else 0. Fast (very simple), works
well

Why even use activation? ==> Linear activation functions results in the output
being linear as well. We need some kind of nonlinear activation function. The
type of non-linear activation function is less important apparently. 
"""
import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_nuerons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_nuerons)
        self.biases = np.zeros((1, n_nuerons))


    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


############### Now we put it work a bit ######################

# Layer_Dense value 1 must be the number of inputs, second value can be anything
layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)
