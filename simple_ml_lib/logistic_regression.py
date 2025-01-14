import numpy as np
from loss_function import CrossEntropyLoss
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_regression(X, Y, learning_rate, epoch):
