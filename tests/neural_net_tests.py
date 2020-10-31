import numpy as np
import matplotlib.pyplot as plt
from mllib.neuralnet.perceptron import Perceptron
from mllib.neuralnet.multi_layer import MultiLayerPerceptron
from mllib.neuralnet.radial_basis import RadialBasisFunction


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


def test_linear_mlp():
    mlp = MultiLayerPerceptron(out_type='linear', epochs=15001, n_hidden=5)

    x = np.ones((1, 40)) * np.linspace(0, 1, 40)
    t = np.sin(2 * np.pi * x) + np.cos(4 * np.pi * x) + np.random.randn(40) * 0.2
    x = x.T
    t = t.T

    train = x[0::2, :]
    test = x[1::4, :]
    valid = x[3::4, :]
    traintar = t[0::2, :]
    testtar = t[1::4, :]
    validtar = t[3::4, :]

    plt.scatter(train, traintar, s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    mlp.fit(train, traintar)


def test_linear_mlp_early_stopping():
    mlp = MultiLayerPerceptron(out_type='linear', epochs=15001, n_hidden=5, early_stopping=True, beta=1.5)

    x = np.ones((1, 100)) * np.linspace(0, 1, 100)
    t = np.sin(2 * np.pi * x) + np.cos(4 * np.pi * x) + np.random.randn(100) * 0.2
    x = x.T
    t = t.T

    train = x[0::2, :]
    test = x[1::4, :]
    valid = x[3::4, :]
    traintar = t[0::2, :]
    testtar = t[1::4, :]
    validtar = t[3::4, :]

    plt.scatter(train, traintar, s=10)
    plt.title('Training')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    mlp.fit(train, traintar)
    predictions = mlp.predict(test)
    print("Test Error: ", 0.5 * np.sum((predictions - testtar) ** 2))

    ax = plt.gca()
    plt.title('Testing')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.scatter(test, testtar, color="b")  # Actual value in blue
    ax.scatter(test, predictions, color="r")  # Predicted value in red
    plt.show()


def test_logistic_mlp():
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [[0], [1], [1], [0]]

    mlp = MultiLayerPerceptron(out_type='logistic', epochs=5001, n_hidden=4)
    mlp.fit(inputs, labels)
    print(mlp.predict(inputs))


def test_logistic_mlp_es():  # Doesn't work
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([[0], [1], [1], [0]])

    mlp = MultiLayerPerceptron(out_type='logistic', epochs=5001, n_hidden=4, early_stopping=True)
    mlp.fit(inputs, labels)
    print(mlp.predict(inputs))


def test_rbf():
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([[0], [1], [1], [1]])
    print(range(np.shape(inputs)[0]))
    rbf = RadialBasisFunction()
    rbf.fit(inputs, labels)


# test_perceptron_on_OR()
# perceptron_test_with_2_neurons()
# test_linear_mlp()
# test_linear_mlp_early_stopping()
# test_logistic_mlp()
# test_logistic_mlp_es()
test_rbf()
