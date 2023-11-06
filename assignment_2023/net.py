from typing import Callable

import numpy as np

from .activation import Sigmoid
from .data_loader import DataLoader
from .linear import Linear
from .loss import CrossEntropyLoss
from .optim import SGD
from .tensor import Tensor


class Net:
    def __init__(self, features: list[int], alpha: float) -> None:
        assert len(features) >= 2
        self.layers: dict[str, Callable] = {}
        for i in range(len(features) - 1):
            self.layers[f"fc{i + 1}"] = Linear(features[i], features[i + 1])
            if i + 1 < len(features) - 1:
                self.layers[f"af{i + 1}"] = Sigmoid(alpha)

        w: Tensor = None
        for layer in self.layers.values():
            if hasattr(layer, "w"):
                w = Tensor(layer.w, w)
        self.w = w

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers.values():
            x = layer(x)
        return x

    def parameters(self) -> Tensor:
        return self.w

    def calc_acc(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        y_label = np.argmax(y, axis=1)
        return np.sum(y_label == t) / len(t)
