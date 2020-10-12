import numpy as np


def coef_determination(y_true, y_pred):
    return 1 - ((np.sum((y_pred - y_true) ** 2)) / (np.sum((y_true - np.mean(y_true)) ** 2)))


def mean_square_error(y_true, y_pred):
    return np.sqrt(np.sum((y_pred - y_true) ** 2) / len(y_true)).round(4)


def sum_of_squares(y_true, y_pred):
    return np.sum((y_pred - y_true) ** 2)
