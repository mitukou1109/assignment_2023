import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from . import net as nn

batch_size = 512
features = [28 * 28, 1024, 512, 10]
learning_rate = 0.5
epochs = 30

seed = 13
torch.manual_seed(seed)

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
train_loader = nn.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
test_loader = nn.DataLoader(test_dataset, batch_size, num_workers=2)

net = nn.Net(features)
optimizer = nn.SGD(net.parameters(), learning_rate)
criterion = nn.CrossEntropyLoss()

avg_acc_list = []
avg_loss_list = []
for i in range(epochs):
    print(f"epoch {i + 1}")
    acc_list = []
    loss_list = []

    x: torch.Tensor
    t: torch.Tensor
    for x, t in train_loader:
        optimizer.zero_grad()
        x = x.reshape(-1, features[0])
        y = net(x)
        acc = net.calc_acc(y, t)
        loss: torch.Tensor = criterion(y, t)
        acc_list.append(acc)
        loss_list.append(loss)
        loss.backward()
        optimizer.step()

    avg_acc = torch.mean(torch.stack(acc_list))
    avg_loss = torch.mean(torch.stack(loss_list))
    print(f"average accuracy: {avg_acc}, average loss: {avg_loss}\n")
    avg_acc_list.append(avg_acc)
    avg_loss_list.append(avg_loss)

plt.figure()
plt.plot(avg_acc_list, label="accuracy")
plt.plot(avg_loss_list, label="loss")
plt.legend()
plt.show()

acc_list = []
print("test")
for x, t in test_loader:
    x = x.reshape(-1, features[0])
    y = net(x)
    acc = net.calc_acc(y, t)
    acc_list.append(acc)

avg_acc = torch.mean(torch.stack(acc_list))
print(f"average accuracy: {avg_acc}")
