import numpy as np

from .tensor import Tensor


class Sigmoid:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def __call__(self, input: Tensor) -> Tensor:
        return Tensor(
            np.exp(np.minimum(input, 0)) / (1 + np.exp(-self.alpha * np.abs(input))),
            input.prev,
        )
