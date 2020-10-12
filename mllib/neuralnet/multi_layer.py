import numpy as np


class MultiLayerPerceptron:
    def __init__(self, n_hidden=2, out_type='logistic', epochs=1000, beta=1, momentum=0.9, eta=0.2,
                 early_stopping=False):
        self.out_weights = 0
        self.hidden_weights = 0
        self.hidden = 0
        self.outputs = 0

        self.n_hidden = n_hidden
        self.out_type = out_type

        self.epochs = epochs
        self.beta = beta
        self.momentum = momentum
        self.eta = eta
        self.early_stopping = early_stopping

        if self.early_stopping:
            self.epochs = 100

    def fit(self, inputs, targets):
        num_in = np.shape(inputs)[1]
        num_out = np.shape(targets)[1]
        self.hidden_weights = (np.random.rand(num_in + 1, self.n_hidden) - 0.5) * 2 / np.sqrt(num_in)
        self.out_weights = (np.random.rand(self.n_hidden + 1, num_out) - 0.5) * 2 / np.sqrt(self.n_hidden)

        self.train(inputs, targets)

    def train_early_stopping(self, inputs, targets, valid, valid_targets):
        pass

    def train(self, inputs, targets):
        num_data = np.shape(inputs)[0]
        inputs = np.concatenate((inputs, -np.ones((num_data, 1))), axis=1)
        out_delta = 0

        update_hidden = np.zeros(np.shape(self.hidden_weights))
        update_out = np.zeros(np.shape(self.out_weights))

        for e in range(self.epochs):
            self.outputs = self.predict(inputs)

            error = 0.5 * np.sum((self.outputs - targets) ** 2)
            if np.mod(e, 50) == 0:
                print("Epoch: ", e, "Error: ", error)

            if self.out_type == 'linear':
                out_delta = (self.outputs - targets) / num_data
            elif self.out_type == 'logistic':
                out_delta = self.beta * (self.outputs - targets) * self.outputs * (1.0 - self.outputs)
            elif self.out_type == 'softmax':
                out_delta = (self.outputs - targets) * (self.outputs * (-self.outputs) + self.outputs) / num_data
            else:
                print('error')

            hidden_delta = self.hidden * self.beta * (1.0 - self.hidden) * np.dot(out_delta, np.transpose(self.out_weights))

            self.hidden_weights -= self.eta * (
                np.dot(np.transpose(inputs), hidden_delta[:, :-1])) + self.momentum * update_hidden
            self.out_weights -= self.eta * (
                np.dot(np.transpose(self.hidden), out_delta)) + self.momentum * update_out

    def predict(self, inputs):
        self.hidden = np.dot(inputs, self.hidden_weights)
        self.hidden = 1.0 / (1.0 + np.exp(-self.beta * self.hidden))
        self.hidden = np.concatenate((self.hidden, -np.ones((np.shape(inputs)[0], 1))), axis=1)

        outputs = np.dot(self.hidden, self.out_weights)

        if self.out_type == 'linear':
            return outputs
        elif self.out_type == 'logistic':
            return 1.0 / (1.0 + np.exp(-self.beta * outputs))
        elif self.out_type == 'softmax':
            normalisers = np.sum(np.exp(outputs), axis=1) * np.ones((1, np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs)) / normalisers)
        else:
            print("error")
