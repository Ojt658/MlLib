from mllib.probabilistic._bayes_linear_regression import BLinReg
from mllib.probabilistic._bayes_logistic_regression import BLogReg
from mllib.probabilistic._naive_bayes import NaiveBayes


class BayesLinearRegression(BLinReg):
    def __init__(self):
        super().__init__()

    def fit(self, inputs, targets):
        super().fit(inputs, targets)

    def predict(self, inputs):
        return super().predict(inputs)


class BayesLogisticRegression(BLogReg):
    def __init__(self):
        super().__init__()


class NaiveBayesClassifer(NaiveBayes):
    def __init__(self):
        super().__init__()
