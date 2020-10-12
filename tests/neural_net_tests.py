import numpy as np
import matplotlib.pyplot as plt
from mllib.neuralnet.perceptron import Perceptron
from mllib.neuralnet.multi_layer import MultiLayerPerceptron


def test_perceptron_on_OR():
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [[0], [1], [1], [1]]

    pcn = Perceptron()
    pcn.fit(inputs, labels)


def perceptron_test_with_2_neurons():
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [[0], [1], [1], [1]]
    pcn = Perceptron(num_nodes=2)
    pcn.fit(inputs, labels)


def test_mlp():
    mlp = MultiLayerPerceptron(out_type='linear', epochs=1001, n_hidden=2)

    x = np.ones((1, 40)) * np.linspace(0, 1, 40)
    t = np.sin(2*np.pi*x) + np.cos(4*np.pi*x) + np.random.randn(40)*0.2
    x = x.T
    t = t.T

    train = x
    # test = x[1::4, :]
    # valid = x[3::4, :]
    traintar = t
    # testtar = t[1::4, :]
    # validtar = t[3::4, :]

    plt.scatter(train, traintar, s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    mlp.fit(train, traintar)

# test_perceptron_on_OR()
# perceptron_test_with_2_neurons()
test_mlp()
