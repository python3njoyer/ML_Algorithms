import math
import numpy as np


def sigmoid(z):
    """
    runs input through sigmoid function and returns output for logistic regression
    """

    g = (1 + math.e ** -z) ** -1

    return g


def compute_cost(X, y, w, b):
    """
    computes a cost function for logistic regression, returns the total cost as a scalar value
    """

    # dimensions for the rows and columns of matrix X
    m, n = X.shape

    # compute vector of inputs for sigmoid function
    z = np.dot(X, w) + b
    h = sigmoid(z)

    # calculate loss based on hypothesized value of each training data
    loss = np.subtract(np.dot(-y, np.log(h)), np.dot((1 - y), np.log(1 - h)))

    # average loss
    total_cost = np.sum(loss) / m


    return total_cost


def compute_gradient(X, y, w, b, lambda_=None):
    """
    finds gradient for gradient descent (logistic regression)
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)  # initialize matrix for gradients of model weights
    dj_db = 0.


    z = np.dot(X, w) + b
    h = sigmoid(z)

    # calculates average gradient for each parameter
    for i in range(m):
        dj_db += (h[i] - y[i])

        for j in range(n):
            dj_dw[j] += (h[i] - y[i]) * X[i, j]

    dj_dw = dj_dw / m
    dj_db = dj_db / m


    return dj_db, dj_dw


def predict(X, w, b):
    """
    predict positive or negative case for binary classification, with threshold = 0.5
    """
    m, n = X.shape  # matrix dimensions
    p = np.zeros(m)  # matrix of zeros for storing probabilities

    z = np.dot(X, w) + b

    # input predicted values into sigmoid to return probability values
    h = sigmoid(z)

    p = (h >= 0.5)


    return p