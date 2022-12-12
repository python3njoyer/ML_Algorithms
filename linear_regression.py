import numpy as np


def compute_cost(x, y, w, b):
    """
    calculates the cost function for linear regression given an input vector x, a label vector y,
    and initial model parameters w and b
    """

    # number of training examples in vector x
    m = x.shape[0]

    # finds hypothesized value with linear equation based on given parameters
    h = np.dot(x, w) + b
    cost = (h - y) ** 2

    # avg. SSE between hypothesized and actual value
    total_cost = np.sum(cost) / (2 * m)


    return total_cost


def compute_gradient(x, y, w, b):
    """
    calculates gradient of parameters w and b for steps in gradient descent
     """

    # num. training examples
    m = x.shape[0]

    # finds gradient for parameters w and b based on each training examples
    h = np.dot(x, w) + b
    grad_b = h - y
    grad_w = (h - y) * x

    # returns total gradient update for each parameter
    dj_db = np.sum(grad_b) / m
    dj_dw = np.sum(grad_w) / m


    return dj_dw, dj_db