import numpy as np


def softmax(input: np.ndarray) -> np.ndarray:
    return np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)


def cross_entropy(predict: np.ndarray, train: np.ndarray) -> np.ndarray:
    epsilon = 1e-7
    return -np.sum(train * np.log(predict + epsilon)) / predict.shape[0]
