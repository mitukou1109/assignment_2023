from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from . import net as nn
from . import transforms

epochs = 300

batch_size = 512
hidden_layer_features = [10]
learning_rate = 0.5
noise_prob = 0
seed = 13

train = True
show_data_sample = True

result = np.ndarray((epochs, 4))
result[:, 0] = np.arange(epochs) + 1
starting_epoch = 0 if train else epochs - 1

file_basename = datetime.now().strftime("%Y%m%d_%H%M%S")

log_header = f"Batch size: {batch_size}, Hidden layer features: {hidden_layer_features}, Learning rate: {learning_rate}, Noise: {int(noise_prob * 100)}%"
log_header = "Epoch, Train loss, Train accuracy, Test accuracy"

np.random.seed(seed)
torch.manual_seed(seed)

dataset_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(
            partial(transforms.random_noise, prob=noise_prob)
        ),
    ]
)
train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=dataset_transform,
    download=True,
)
test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=dataset_transform,
    download=True,
)
train_loader = nn.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
test_loader = nn.DataLoader(test_dataset, batch_size, num_workers=2)
if show_data_sample:
    sample = next(iter(train_loader))[0][0].permute(1, 2, 0)
    plt.imshow(sample, cmap="gray_r")
    plt.axis("off")
    plt.show()

input_rows = 28
input_cols = 28
output_features = 10

net = nn.Net([input_rows * input_cols] + hidden_layer_features + [output_features])
optimizer = nn.SGD(net.parameters(), learning_rate)
criterion = nn.CrossEntropyLoss()

train_loss = result[:, 1]
train_acc = result[:, 2]
test_acc = result[:, 3]

for i in np.arange(starting_epoch, epochs):
    if train:
        print(f"epoch {i + 1}")

        acc = 0
        loss = 0
        x: torch.Tensor
        for x, t in train_loader:
            optimizer.zero_grad()
            x = x.reshape(-1, input_rows * input_cols)
            y = net(x)
            acc += net.calc_acc(y, t)
            loss: torch.Tensor = criterion(y, t)
            loss.backward()
            optimizer.step()

        train_acc[i] = acc / len(train_loader)
        train_loss[i] = loss / len(train_loader)

    acc = 0
    for x, t in test_loader:
        x = x.reshape(-1, input_rows * input_cols)
        y = net(x)
        acc += net.calc_acc(y, t)

    test_acc[i] = acc / len(test_loader)

    print(
        (
            f"train loss: {train_loss[i]}, train accuracy: {train_acc[i]}, "
            if train
            else ""
        )
        + f"test accuracy: {test_acc[i]}",
        end="\n\n",
    )

    if train:
        np.savetxt(
            f"log/{file_basename}.csv",
            result[: i + 1],
            fmt=["%d", "%f", "%f", "%f"],
            delimiter=",",
            header=log_header,
        )
