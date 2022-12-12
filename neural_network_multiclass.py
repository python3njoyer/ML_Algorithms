import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid


# construct three-layer model using ReLU activation function for hidden layers
model = Sequential(
    [
        tf.keras.Input(shape=(400,)),
        Dense(units=25, activation='relu'),
        Dense(units=15, activation='relu'),
        Dense(units=10, activation='linear')

    ], name="my_model"
)


def my_softmax(z):
    """
    implementation of softmax function to return vector of probabilities of each class, summing to one.
    z: vector from output layer of model above
    """

    size = len(z)
    a = np.zeros(size)
    e = np.exp(1)

    ez_sum = 0
    for k in range(size):
        ez_sum += e ** z[k]
    for j in range(size):
        a[j] = e ** z[j] / ez_sum

    # returns vector of probabilities that training example is each class
    return a