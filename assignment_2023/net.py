import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader


class Net(torch.nn.Module):
    def __init__(self, features: list[int]) -> None:
        assert len(features) >= 2
        super(Net, self).__init__()
        self.layers = torch.nn.Sequential()
        for i in range(len(features) - 1):
            self.layers.add_module(
                f"fc{i + 1}", torch.nn.Linear(features[i], features[i + 1])
            )
            if i + 1 < len(features) - 1:
                self.layers.add_module(f"af{i + 1}", torch.nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def calc_acc(self, y: torch.Tensor, t: torch.Tensor) -> float:
        y_label = torch.argmax(y, dim=1)
        return torch.sum(y_label == t) / len(t)
