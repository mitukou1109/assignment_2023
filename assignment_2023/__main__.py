from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from . import net as nn
from . import transforms

batch_size = 512
features = [28 * 28, 1024, 512, 10]
alpha = 1.0
learning_rate = 0.5
epochs = 10

show_data_sample = True
noise_prob = 0.25

show_loss = True
show_learning_curve = True

seed = 13
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
    sample: np.ndarray = next(iter(train_loader))[0][0].transpose(1, 2, 0)
    plt.imshow(sample)
    plt.show()

net = nn.Net(features, alpha)
optimizer = nn.SGD(net.parameters(), learning_rate)
criterion = nn.CrossEntropyLoss(net.parameters(), alpha)

train_acc = []
train_loss = []
test_acc = []

for i in range(epochs):
    print(f"epoch {i + 1}")

    acc = 0
    loss = 0
    for x, t in train_loader:
        x = x.reshape(-1, features[0])
        y = net(x)
        acc += net.calc_acc(y, t)
        loss += criterion(y, t)
        grad = criterion.backward()
        optimizer.step(grad)

    train_acc.append(acc / len(train_loader))
    train_loss.append(loss / len(train_loader))

    acc = 0
    for x, t in test_loader:
        x = x.reshape(-1, features[0])
        y = net(x)
        acc += net.calc_acc(y, t)

    test_acc.append(acc / len(test_loader))

    print(
        f"train accuracy: {train_acc[-1]}, train loss: {train_loss[-1]}, test accuracy: {test_acc[-1]}\n"
    )

x_ticks = np.arange(1, epochs + 1)
title = f"Batch size: {batch_size}, Hidden layers: {features[1:-1]}, \nLearning rate: {learning_rate}, Noise: {int(noise_prob * 100)}%"

if show_loss:
    loss_curve = plt.figure(num="Loss")
    ax = loss_curve.add_subplot()
    ax.set_title(title)
    ax.plot(x_ticks, train_loss)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_xticks(x_ticks)
    ax.grid(axis="y")

if show_learning_curve:
    learning_curve = plt.figure(num="Learning curve")
    ax = learning_curve.add_subplot()
    ax.set_title(title)
    ax.plot(x_ticks, train_acc, label="train")
    ax.plot(x_ticks, test_acc, label="test")
    ax.legend(loc="lower right")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_xticks(x_ticks)
    y_min = np.round(min(train_acc + test_acc), 1)
    ax.set_yticks(np.arange(y_min, 1.01, 0.05 if 1 - y_min < 0.4 else 0.1))
    ax.set_ylim(y_min - (1 - y_min) * 0.05, 1 + (1 - y_min) * 0.05)
    ax.grid(axis="y")

plt.show()
