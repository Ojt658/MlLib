from mllib.linear.linear_regression import LinearRegression1, LinearRegression
from mllib.metrics.regression import mean_square_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def test_linear_regression():
    lr = LinearRegression1()

    np.random.seed(0)
    x_train = np.random.rand(100, 1)
    y = 2 + 3 * x_train + np.random.rand(100, 1)

    plt.scatter(x_train, y, s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    lr.fit(x_train, y)
    np.random.seed(1)
    x_test = np.random.rand(100, 1)
    y_test = 2 + 3 * x_test + np.random.rand(100, 1)
    predictions = lr.predict(x_test)
    print(mean_square_error(y_test, predictions))
    print(lr.predict(0.5))


def test_with_2d_data():
    x = np.random.random((100, 5))
    y = 2 + 3 * x[0][0] + np.random.rand(100, 1)

    lr = LinearRegression1()
    lr.fit(x[:80], y[:80])

    predictions = lr.predict(x[81:])
    print(mean_square_error(y[81:], predictions))


def test_with_df():
    a = np.random.rand(3, 100)

    dict4df = {
        'a': [0, 1, 2, 3, 4, 5],
        'b': [0, 1, 2, 3, 4, 5],
        'c': [0, 1, 2, 3, 4, 5]
    }

    print(dict4df)
    x = pd.DataFrame(data=dict4df)
    y = [0, 1, 2, 3, 4, 5]

    lr = LinearRegression1()
    lr.fit(x, y)


def test_sec_alg():
    lr = LinearRegression()

    np.random.seed(0)
    x_train = np.random.rand(100, 1)
    y = 2 + 3 * x_train + np.random.rand(100, 1)

    plt.scatter(x_train, y, s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    lr.fit(x_train, y)
    np.random.seed(1)
    x_test = np.random.rand(100, 1)
    y_test = 2 + 3 * x_test + np.random.rand(100, 1)
    predictions = lr.predict(x_test)
    print(mean_square_error(y_test, predictions))
    print(lr.predict([[0.5]]))


def test_sec_alg_with_2d():
    x = np.random.random((100, 5))
    y = 2 + 3 * x[0][0] + np.random.rand(100, 1)

    lr = LinearRegression()
    lr.fit(x[:80], y[:80])

    predictions = lr.predict(x[81:])
    print(mean_square_error(y[81:], predictions))


# test_linear_regression()
# test_with_2d_data()
# test_with_df()
# test_sec_alg()
test_sec_alg_with_2d()
