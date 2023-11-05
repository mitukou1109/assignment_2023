import numpy as np

from .single_neuron import SingleNeuron
from .tensor import Tensor


class Linear:
    def __init__(self, in_features: int, out_features: int) -> None:
        self.out_features = out_features
        self.neurons = [SingleNeuron() for _ in range(out_features)]
        self.w = np.random.rand(out_features, in_features + 1) * 2 - 1

    def __call__(self, x: Tensor) -> Tensor:
        y = Tensor(np.zeros((x.shape[0], self.out_features)), x)

        for i in range(self.out_features):
            self.neurons[i].w = self.w[i]
            y[:, i] = self.neurons[i](x)

        return y
