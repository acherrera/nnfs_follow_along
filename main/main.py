import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data

nnfs.init()
plt.style.use('dark_background')

class Layer_Dense:
    def __init__(self, n_inputs, n_nuerons):
        """
        Sets the initial weights and the biases for the given layer. This is a dense layer because all of 
        the output values are passed to the input values of the next layer
        n_input - expected number of inputs. Can be from inputs or previous layer
        n_nuerons - number of nuerons to use in the layer
        """
        self.weights = 0.10 * np.random.randn(n_inputs, n_nuerons)
        self.biases = np.zeros((1, n_nuerons))


    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def backwards(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Apply gradients to values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_RelU:
    def __init__(self):
        """
        Activates the nuerons
        """
        pass
    
    def forward(self, inputs):
        self.inputs = inputs 
        self.output = np.maximum(0, inputs)

    def backwards(self, dvalues):
        # Copy the variables before moding it
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <=0] = 0


class Activation_Softmax:
    def __init__(self):
        """
        Cleans up the output values so they are uniform without throwing out negative values.
        """
        pass
    
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def __init__(self):
        """
        Calculates the data aand regularization losses given model output 
        and ground truth values. This is a common class that will be inherited by later loss classes
        to build upon
        """
        pass
    def calculate(self, output, y):
        """
        output: output of the model
        y: ground truth values
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # clip to prevent log(0) error. Both sides to prevent value biasing
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        # Checking for categorical labels vs one-hot encoded
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
            range(samples),
            y_true
            ]
            # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
            y_pred_clipped*y_true,
            axis=1
            )
        
        negative_log = -np.log(correct_confidences)
        return negative_log


X, y = vertical_data(samples=100, classes=3)

# Input Layer - 2 inputs (x,y) and 3 outputs
dense1 = Layer_Dense(2,3)
activation1 = Activation_RelU()

# Layer 2 - 3 input and 3 output. Want to guess color (class) based on x,y values
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossEntropy()

# Now we want to keep track of what we are using
lowest_loss = 999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()


for iteration in range(10000):

    dense1.weights += 0.05*np.random.randn(2,3)
    dense1.biases += 0.05*np.random.randn(1,3)
    dense2.weights += 0.05*np.random.randn(3,3)
    dense2.biases += 0.05*np.random.randn(1,3)

    # Run layer1
    dense1.forward(X)
    activation1.forward(dense1.output)
    # Run layer2
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Calculate loss, accuracy
    loss = loss_function.calculate(activation2.output, y)
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y) # Works because y is one of (0,1,2)

    if loss < lowest_loss:
        # Update with new values
        print(f"New values --> Iteration: {iteration}\tLoss: \
        {loss}\tAccuracy: {accuracy}")
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
        best_accuracy = accuracy
        best_iteration = iteration

    else:
        # Reset values to best values
        dense1.weights = best_dense1_weights
        dense1.biases = best_dense1_biases
        dense2.weights = best_dense2_weights
        dense2.biases = best_dense2_biases

print("FINAL RESULTS: ")
print(f"New values --> Iteration: {best_iteration}\tLoss: \
{lowest_loss}\tAccuracy: {best_accuracy}")
