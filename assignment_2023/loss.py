from typing import Self

import numpy as np

from .functional import cross_entropy, softmax
from .tensor import Tensor


class CrossEntropyLoss:
    def __init__(self, params: Tensor, alpha: float) -> None:
        self.loss = None
        self.params = params
        self.alpha = alpha

    def __call__(self, y: Tensor, t: np.ndarray) -> Self:
        self.y = y
        self.y_prob = softmax(y)
        self.t_prob = np.zeros(self.y_prob.shape)
        self.t_prob[np.arange(t.shape[0]), t] = 1
        self.loss = cross_entropy(self.y_prob, self.t_prob)
        return self

    def item(self) -> np.ndarray:
        return self.loss

    def backward(self) -> Tensor:
        # assuming activation function is softmax for output layer, sigmoid for others
        delta = self.y_prob - self.t_prob
        x: Tensor = self.y.prev
        grad = Tensor(delta.T @ np.hstack([x, np.ones((x.shape[0], 1))]))
        w = self.params
        g = grad

        while x.prev is not None:
            delta = delta @ w[:, :-1] @ (self.alpha * x.T @ (1 - x))
            x = x.prev
            g.prev = Tensor(delta.T @ np.hstack([x, np.ones((x.shape[0], 1))]))
            w = w.prev
            g = g.prev

        return grad
