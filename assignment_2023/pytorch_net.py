import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.nn.functional import relu
from torch.optim import SGD
from torch.utils.data import DataLoader


class Net(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_1_features: int,
        hidden_2_features: int,
        out_features: int,
    ) -> None:
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden_1_features)
        self.fc2 = torch.nn.Linear(hidden_1_features, hidden_2_features)
        self.fc3 = torch.nn.Linear(hidden_2_features, out_features)

    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def calc_acc(self, y: torch.Tensor, t: torch.Tensor):
        y_label = torch.argmax(y, dim=1)
        return torch.sum(y_label == t) / len(t)
