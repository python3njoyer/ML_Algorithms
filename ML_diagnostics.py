# A collection of diagnostic tools for evaluating machine learning models
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


def eval_mse(y, yhat):
    """
    evaluates the mean squared error for predicted values of a data set
    """
    m = len(y)
    err = 0.0
    for i in range(m):
        err += (yhat[i] - y[i]) ** 2

    err = err / (2 * m)

    return (err)


def eval_cat_err(y, yhat):
    """
    calculates categorization error rate-- % of training examples incorrectly classified by model
    """
    m = len(y)
    incorrect = 0

    for i in range(m):
        incorrect += np.sum(y[i] != yhat[i])

    cerr = incorrect / m

    return (cerr)


# examples of 2 neural networks, one with many parameters and one with few (model overfitting vs. underfitting)

# overfitting:
model = Sequential(
    [
        Dense(units=120, activation='relu'),
        Dense(units=40, activation='relu'),
        Dense(units=6, activation='linear')

    ], name="Complex"
)
# sets loss function and optimizes learning rate with Adam optimization algorithm
model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
)
# fits model to training examples
# model.fit(
#     X, y,
#     epochs=1000
# )


# underfitting:
model_s = Sequential(
    [
       Dense(units=6, activation='relu'),
         Dense(units=6, activation='linear')
    ], name = "Simple"
)
model_s.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
)
# model_s.fit(
#     X,y,
#     epochs=1000
# )


# complex model but with regularization:
model_r = Sequential(
    [
        Dense(units=120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        Dense(units=40, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        Dense(units=6, activation='linear')
    ], name= None
)
model_r.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
)
# model_r.fit(
#     X,y,
#     epochs=1000
# )