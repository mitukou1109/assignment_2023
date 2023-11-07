import numpy as np

from .tensor import Tensor


class Linear:
    def __init__(self, in_features: int, out_features: int) -> None:
        self.w = np.random.rand(out_features, in_features + 1) * 2 - 1

    def __call__(self, x: Tensor) -> Tensor:
        return Tensor(np.hstack([x, np.ones((x.shape[0], 1))]) @ self.w.T, x)
