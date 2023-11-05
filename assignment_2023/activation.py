import numpy as np

from .tensor import Tensor


class Sigmoid:
    def __call__(self, input: Tensor) -> Tensor:
        return Tensor(
            np.where(
                input > 0,
                1 / (1 + np.exp(-input)),
                np.exp(input) / (1 + np.exp(input)),
            ),
            input.prev,
        )
