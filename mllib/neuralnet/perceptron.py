import numpy as np


class Perceptron:
    def __init__(self, num_nodes=1, n_iterations=100, eta=0.05):
        self.T = n_iterations
        self.eta = eta
        self.N = num_nodes
        self.weights = 0

    def fit(self, inputs, labels):
        inputs = np.concatenate((inputs, -np.ones((len(inputs), self.N))), axis=1)
        labels = np.array(labels)
        self.weights = .2 * np.random.random((len(inputs[0]), self.N)) - 0.5
        for t in range(self.T):
            predictions = self.predict(inputs)
            print("Iteration: " + str(t))
            print(self.weights)
            print(predictions)
            if not (np.all([labels[x] == predictions[x] for x in labels])):
                self.weights -= self.eta * np.dot(np.transpose(inputs), predictions-labels)
            else:
                break

    def predict(self, inputs):
        activations = np.dot(inputs, self.weights)
        return np.where(activations > 0, 1, 0)
