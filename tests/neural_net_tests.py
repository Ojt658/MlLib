import numpy as np
from mllib.neuralnet.perceptron import Perceptron

def test_perceptron_on_OR():
    inputs = [[0,0], [0,1], [1,0], [1,1]]
    labels = [[0], [1], [1], [1]]

    pcn = Perceptron()
    pcn.fit(inputs, labels)

def perceptron_test_with_2_neurons():
    inputs = [[0,0], [0,1], [1,0], [1,1]]
    labels = [[0], [1], [1], [1]]
    pcn = Perceptron(num_nodes=2)
    pcn.fit(inputs, labels)

test_perceptron_on_OR()
perceptron_test_with_2_neurons()