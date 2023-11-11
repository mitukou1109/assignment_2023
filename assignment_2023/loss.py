import numpy as np

from .functional import cross_entropy, softmax
from .tensor import Tensor


class CrossEntropyLoss:
    def __init__(self, params: Tensor) -> None:
        self.params = params

    def __call__(self, y: Tensor, t: np.ndarray) -> float:
        self.y_prob = softmax(y)
        self.t_prob = np.zeros(self.y_prob.shape)
        self.t_prob[np.arange(t.shape[0]), t] = 1
        return cross_entropy(self.y_prob, self.t_prob)

    def backward(self) -> Tensor:
        # assuming activation function is softmax for output layer, sigmoid for others
        delta = self.y_prob - self.t_prob
        x: Tensor = self.y_prob.prev
        grad = Tensor(delta.T @ np.hstack([x, np.ones((x.shape[0], 1))]) / x.shape[0])
        w = self.params
        g = grad

        while x.prev is not None:
            delta = (delta @ w[:, :-1]) * (x * (1 - x))
            x = x.prev
            g.prev = Tensor(
                delta.T @ np.hstack([x, np.ones((x.shape[0], 1))]) / x.shape[0]
            )
            w = w.prev
            g = g.prev

        return grad
