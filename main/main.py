import numpy as np
import nnfs
import matplotlib.pyplot as plt
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
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Apply gradients to values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def __init__(self):
        """
        Activates the nuerons
        """
        pass

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
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
        # Keep for later
        self.inputs = inputs

        # Reduce all by largest value, take log, normalize values
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1 , keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


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
    def __init__(self):
        pass


    def forward(self, y_pred, y_true):
        """
        Run values forward to calculate loss
        Args:
            y_pred: predicated values
            y_true: actual values
        """
        samples = len(y_pred)
        # clip to prevent log(0) error. Both sides to prevent value biasing
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # Checking for categorical labels vs one-hot encoded
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
            range(samples),
            y_true
            ]

        elif len(y_true.shape) == 2:
            # Mask values - only for one-hot encoded labels
            correct_confidences = np.sum(
            y_pred_clipped*y_true,
            axis=1
            )

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backwards(self, dvalues, y_true):
        """
        Run the derivative values backwards
        Args:
            dvalues: derivative values given
            y_true: ground truth values
        """
        samples = len(dvalues)
        labels = len(dvalues[0])

        # discrete => one-hot encoded
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Gradient calculation and normalize
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy():

    def __init__(self):
        """
        Combined Softmax activation and cross-entropy because the backward step is WAY faster this way
        Allows us to run the forward step of the other classes and define our own backwards
        """
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    # Forward pass
    def forward(self, inputs, y_true):
        """
        Perform the forward softmax / loss calculation
        Args:
            inputs: array of inputs
            y_true: array of ground truth values
        """

        # Do activation, calculate output, return
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        """
        Run the functions backwards which resultsi in y-hat - y as shown in the derivation section of the NNFS book
        Args:
            dvalues: derivative values to feed in
            y_true: ground truth array
        """
        samples = len(dvalues)
        # One-hot => descrete conversion. Will return array of where the argmax is located for each sample
        #[[1,0,0],[0,0,1],[0,1,0]] => [0,2,1]
        if len(y_true.shape)==2:
            y_true = np.argmax(y_true, axis=1)

        # I do not understand how this part is working. An example would be wonderful. This is taking advantage of
        # one-hot encoding to make it work, but I don't know what "self.dinputs" looks like
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        """
        Intialize the model and set the learning rate
        """
        self.learning_rate = learning_rate

    def update_params(self, layer):
        """
        Use the larning rate the update the weights
        """
        # Update parameters
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases



if __name__ == "__main__":

    import nnfs
    from nnfs.datasets import spiral_data
    nnfs.init()

    ################### Intialize  ###########################
    X, y = spiral_data(samples=100, classes=3)
    # Create Dense layer with 2 input features and 64 output values
    dense1 = Layer_Dense(2, 64)
    # Create ReLU activation (to be used with Dense layer):
    activation1 = Activation_ReLU()
    # Create second Dense layer with 64 input features (as we take output
    # of previous layer here) and 3 output values (output values)
    dense2 = Layer_Dense(64, 3)
    # Create Softmax classifier's combined loss and activation
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    # Create optimizer
    optimizer = Optimizer_SGD()


    ######################## Forward Pass #######################
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)
    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)
    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)
    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = loss_activation.forward(dense2.output, y)
    # Let's print loss value
    print('loss:', loss)


    ################### Test predictions #####################
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions==y)
        print('acc:', accuracy)

    ############# Backward pass  ##########################
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    ############# Update weights and biases ##################
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
