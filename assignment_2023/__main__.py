import numpy as np
import torchvision

from . import pytorch_net as nn

batch_size = 16
hidden_features = [1024, 512]
out_features = 10
learning_rate = 0.01

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
train_loader = nn.DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = nn.DataLoader(test_dataset, batch_size)

in_features = train_dataset[0][0].numel()
net = nn.Net(in_features, *hidden_features, out_features)
optimizer = nn.SGD(net.parameters(), learning_rate)
criterion = nn.CrossEntropyLoss()

for x, t in train_loader:
    optimizer.zero_grad()
    x = x.view(-1, in_features)
    y = net(x)
    acc = net.calc_acc(y, t)
    loss = criterion(y, t)
    print(f"accuracy: {acc}, loss: {loss}")
    loss.backward()
    optimizer.step()

accs = []
for x, t in test_loader:
    x = x.view(-1, in_features)
    y = net(x)
    acc = net.calc_acc(y, t)
    accs.append(acc)

avg_acc = np.mean(accs)
print(f"average accuracy: {avg_acc}")
