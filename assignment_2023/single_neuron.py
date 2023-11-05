import numpy as np


class SingleNeuron:
    def __init__(self, in_features: int = -1) -> None:
        self.w = np.random.rand(in_features + 1) * 2 - 1

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.hstack([x, np.ones((x.shape[0], 1))]) @ self.w.T
