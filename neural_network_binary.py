# developed for binary classification of handwritten digits from MNIST dataset

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# builds model with 2 hidden layers, using sigmoid as activation function
model = Sequential(
    [
        tf.keras.Input(shape=(400,)),  # specify input size

        Dense(units=25, activation='sigmoid'),
        Dense(units=15, activation='sigmoid'),
        Dense(units=1, activation='sigmoid')

    ], name="my_model"
)


def my_dense(a_in, W, b, g):
    """
    manual implementation of neural network layer
    param g: activation function
    """
    units = W.shape[1]
    a_out = np.zeros(units)

    for j in range(units):
        z = np.dot(a_in, W[:, j]) + b[j]
        a_out[j] = g(z)

    return (a_out)


def my_dense_v(A_in, W, b, g):
    """
    vectorized implementation of neural network layer using matmul function
    """

    z = np.matmul(A_in, W) + b
    A_out = g(z)

    return (A_out)