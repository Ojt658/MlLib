import numpy as np
from .perceptron import Perceptron

class RadialBasisFunction:
    def __init__(self, eta=0.25, epochs=100, num_rbf=5, normalise=True):
        self.eta = eta
        self.epochs = epochs
        self.num_RBF = num_rbf
        self.rbf_weights = 0
        self.out_weights = 0
        self.hidden = 0
        self.normalise = normalise

    def get_distance(self, centroids, inputs):
        euc_sum = 0
        for c in range(self.num_RBF):
            for x in range(len(inputs)):
                euc_sum += (centroids[c] - inputs[x]) ** 2
        return np.sqrt(euc_sum)

    def fit(self, inputs, targets):
        centroids = range(np.shape(inputs)[0])
        np.random.shuffle(centroids)
        for i in range(self.num_RBF):
            self.rbf_weights[:, i] = inputs[centroids[i], :]

        d = self.get_distance(self.rbf_weights, inputs)
        sigma = d / np.sqrt(2 * self.num_RBF)

        for i in range(self.num_RBF):
            self.hidden[:, i] = np.exp(
                -np.sum((inputs - np.ones((1, np.shape(inputs)[1])) * self.rbf_weights[:, i]) ** 2, axis=1) / (
                            2 * sigma ** 2))
        if self.normalise:
            self.hidden[:, :-1] /= np.transpose(
                np.ones((1, np.shape(self.hidden)[0])) * self.hidden[:, :-1].sum(axis=1))

        # self.hidden = np.concatenate(self.hidden, -np.ones((np.shape(self.hidden)[0], 1)), axis=1)
        self.out_weights = np.dot(np.linalg.pinv(self.hidden), targets)
        print(self.out_weights)

    # def predict(self, inputs):
