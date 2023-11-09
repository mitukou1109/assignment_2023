import numpy as np

from .tensor import Tensor


class Sigmoid:
    def __call__(self, input: Tensor) -> Tensor:
        return Tensor(
            np.exp(np.minimum(input, 0)) / (1 + np.exp(-np.abs(input))),
            input.prev,
        )
