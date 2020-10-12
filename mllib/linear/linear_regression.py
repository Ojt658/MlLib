import numpy as np


class LinearRegression1:
    def __init__(self, eta=0.05, n_iterations=1000):
        self.cost_ = []
        self.eta = eta
        self.n_iterations = n_iterations
        self.w_ = 0

    def fit(self, x, y):
        self.w_ = np.zeros((x.shape[1], 1))
        m = x.shape[0]

        for _ in range(self.n_iterations):
            y_pred = np.dot(x, self.w_)
            residuals = y_pred - y
            gradient_vector = np.dot(x.T, residuals)
            self.w_ -= (self.eta / m) * gradient_vector
            cost = np.sum((residuals ** 2)) / (2 * m)
            self.cost_.append(cost)
        return self

    def predict(self, x):
        return np.dot(x, self.w_).round(4)


class LinearRegression:
    def __init__(self):
        self.beta = 0

    def fit(self, inputs, labels):
        inputs = np.concatenate((inputs, -np.ones((np.shape(inputs)[0], 1))), axis=1)
        self.beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(inputs), inputs)), np.transpose(inputs)), labels)

    def predict(self, inputs):
        inputs = np.concatenate((inputs, -np.ones((np.shape(inputs)[0], 1))), axis=1)
        return np.dot(inputs, self.beta)