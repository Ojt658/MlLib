import numpy as np


class MultiLayerPerceptron:
    def __init__(self, n_hidden=2, out_type='logistic', epochs=1000, beta=1, momentum=0.9, eta=0.25,
                 early_stopping=False):
        # Initialise weights
        self.out_weights = 0
        self.hidden_weights = 0

        # Initialise layers ( later will be matrices )
        self.hidden = 0
        self.outputs = 0

        self.n_hidden = n_hidden  # Num of hidden layers
        self.out_type = out_type  # Output type: logistic, linear or softmax

        self.epochs = epochs  # Num iterations of the algorithm
        # Define learning constants
        self.beta = beta
        self.momentum = momentum
        self.eta = eta
        # If using early stopping
        self.early_stopping = early_stopping
        if self.early_stopping:
            self.epochs = 1000  # Lower epoch as looping over the loop

    def fit(self, inputs, targets):
        num_in = np.shape(inputs)[1]  # Number input nodes
        num_out = np.shape(targets)[1]  # Number output nodes
        # Initialise weight matrices
        self.hidden_weights = (np.random.rand(num_in + 1, self.n_hidden) - 0.5) * 2 / np.sqrt(num_in)
        self.out_weights = (np.random.rand(self.n_hidden + 1, num_out) - 0.5) * 2 / np.sqrt(self.n_hidden)

        # Call the main algorithm and print the final error after they have completed
        if not self.early_stopping:
            self.train(inputs, targets)
            print("Final Error: ", 0.5 * np.sum((self.outputs - targets) ** 2))
        else:
            print("Final Error: ", self.train_early_stopping(inputs, targets))  # Final error on validation set

    def train_early_stopping(self, inputs, targets):
        # Initialise extreme values for the old errors
        old_val1 = 100002
        old_val2 = 100001
        new_val = 100000

        # Create a validation test set to test on data that wasn't used to train
        valid = inputs[1::4, :]
        valid_target = targets[1::4, :]
        inputs = inputs[3::4, :]
        targets = targets[3::4, :]

        # Add the biases to the validation set
        valid = np.concatenate((valid, -np.ones((np.shape(valid)[0], 1))), axis=1)

        while (old_val1 - new_val) > 0.001 or (old_val2 - old_val1) > 0.001:  # While the error is still decreasing
            self.train(inputs, targets)  # Train the model
            # Adjust old error values
            old_val2 = old_val1
            old_val1 = new_val
            validout = self._predict(valid)  # Predict on the validation set
            new_val = 0.5 * np.sum((valid_target - validout) ** 2)  # Calculate the new error on the validation set

        print("Stopped: ", new_val, old_val1, old_val2)
        return new_val

    def train(self, inputs, targets):
        num_data = np.shape(inputs)[0]  # Number of input data rows
        inputs = np.concatenate((inputs, -np.ones((num_data, 1))), axis=1)  # Add the biases for the hidden layer nodes
        out_delta = 0  # Initialise the var for the output gradient

        # Initialise matrices for updating the weights
        update_hidden = np.zeros(np.shape(self.hidden_weights))
        update_out = np.zeros(np.shape(self.out_weights))

        for e in range(self.epochs):  # Loop over the epochs / iterations
            self.outputs = self._predict(inputs)  # Predict the output values

            error = 0.5 * np.sum(
                (self.outputs - targets) ** 2)  # Calculate sum of the squares error for printing progress
            if np.mod(e, 50) == 0:  # Only print multiples of 50
                print("Epoch: ", e, "Error: ", error)

            # Find gradient of weights depending on output type
            if self.out_type == 'linear':
                out_delta = (self.outputs - targets) / num_data
            elif self.out_type == 'logistic':
                out_delta = self.beta * (self.outputs - targets) * self.outputs * (1.0 - self.outputs)
            elif self.out_type == 'softmax':
                out_delta = (self.outputs - targets) * (self.outputs * (-self.outputs) + self.outputs) / num_data
            else:
                print('error')

            # Find the gradient of the hidden layers
            hidden_delta = self.hidden * self.beta * (1.0 - self.hidden) * np.dot(out_delta,
                                                                                  np.transpose(self.out_weights))

            # Update the class weights according to the direction of the gradients (the deltas)
            self.hidden_weights -= self.eta * (
                np.dot(np.transpose(inputs), hidden_delta[:, :-1])) + self.momentum * update_hidden
            self.out_weights -= self.eta * (
                np.dot(np.transpose(self.hidden), out_delta)) + self.momentum * update_out

    def _predict(self, inputs):
        # Calculate the activations of the hidden layer nodes
        self.hidden = np.dot(inputs, self.hidden_weights)
        self.hidden = 1.0 / (1.0 + np.exp(-self.beta * self.hidden))
        # Add biases to the results for output layer
        self.hidden = np.concatenate((self.hidden, -np.ones((np.shape(inputs)[0], 1))), axis=1)

        # Calculate output activations
        outputs = np.dot(self.hidden, self.out_weights)

        # Choose which activation function to use; depending on mode
        if self.out_type == 'linear':
            return outputs
        elif self.out_type == 'logistic':
            return 1.0 / (1.0 + np.exp(-self.beta * outputs))  # Sigmoid function
        elif self.out_type == 'softmax':
            normalisers = np.sum(np.exp(outputs), axis=1) * np.ones((1, np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs)) / normalisers)
        else:
            print("error")

    def predict(self, inputs):
        # Add biases to the input data from external calls
        inputs = np.concatenate((inputs, -np.ones((np.shape(inputs)[0], 1))), axis=1)
        return self._predict(inputs)
