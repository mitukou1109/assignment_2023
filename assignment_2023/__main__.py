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

noise_prob = 0.25

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

net = nn.Net(features, alpha)
optimizer = nn.SGD(net.parameters(), learning_rate)
criterion = nn.CrossEntropyLoss(net.parameters(), alpha)

avg_acc_list = []
avg_loss_list = []
print("train")
for i in range(epochs):
    print(f"epoch {i + 1}")
    acc_list = []
    loss_list = []

    for x, t in train_loader:
        x = x.reshape(-1, features[0])
        y = net(x)
        acc = net.calc_acc(y, t)
        loss = criterion(y, t)
        acc_list.append(acc)
        loss_list.append(loss)
        grad = criterion.backward()
        optimizer.step(grad)

    avg_acc = np.mean(acc_list)
    avg_loss = np.mean(loss_list)
    avg_acc_list.append(avg_acc)
    avg_loss_list.append(avg_loss)

    print(f"average accuracy: {avg_acc}, average loss: {avg_loss}\n")

fig = plt.figure()
acc_ax = fig.add_subplot(111)
acc_ax.plot(avg_acc_list, label="accuracy", color="C0")
loss_ax = acc_ax.twinx()
loss_ax.plot(avg_loss_list, label="loss", color="C1")
h_acc, l_acc = acc_ax.get_legend_handles_labels()
h_loss, l_loss = loss_ax.get_legend_handles_labels()
acc_ax.legend(h_acc + h_loss, l_acc + l_loss, loc="lower left")
acc_ax.set_xlabel("epoch")
acc_ax.set_ylabel("accuracy")
loss_ax.set_ylabel("loss")
acc_ax.set_ylim(0, 1)
plt.show()

acc_list = []
print("test")
for x, t in test_loader:
    x = x.reshape(-1, features[0])
    y = net(x)
    acc = net.calc_acc(y, t)
    acc_list.append(acc)

avg_acc = np.mean(acc_list)
print(f"average accuracy: {avg_acc}")
