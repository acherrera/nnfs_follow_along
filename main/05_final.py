import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

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


class Activation_RelU:
    def __init__(self):
        """
        Activates the nuerons
        """
        pass
    
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


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


X, y = spiral_data(samples=100, classes=3)

# Input Layer
dense1 = Layer_Dense(2,3)
activation1 = Activation_RelU()

# Layer 2
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

# Run layer1
dense1.forward(X)
activation1.forward(dense1.output)

# Run layer2
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

# Calculate loss
loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)
print(loss)

# Calculate accuracy
predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy}")
