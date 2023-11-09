from functools import partial

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from . import net as nn
from . import transforms

batch_size = 512
features = [28 * 28, 1024, 512, 10]
learning_rate = 0.5
epochs = 100

show_data_sample = False
noise_prob = 0

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

net = nn.Net(features)
optimizer = nn.SGD(net.parameters(), learning_rate)
criterion = nn.CrossEntropyLoss(net.parameters())

result = np.ndarray((epochs, 4))
epochs = np.arange(epochs)
result[:, 0] = epochs + 1
train_loss = result[:, 1]
train_acc = result[:, 2]
test_acc = result[:, 3]

for i in epochs:
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

    train_acc[i] = acc / len(train_loader)
    train_loss[i] = loss / len(train_loader)

    acc = 0
    for x, t in test_loader:
        x = x.reshape(-1, features[0])
        y = net(x)
        acc += net.calc_acc(y, t)

    test_acc[i] = acc / len(test_loader)

    print(
        f"train loss: {train_loss[i]}, train accuracy: {train_acc[i]}, test accuracy: {test_acc[i]}\n"
    )

header = f"Batch size: {batch_size}, Hidden layers: {features[1:-1]}, Learning rate: {learning_rate}, Noise: {int(noise_prob * 100)}%"
header += "\nEpoch, Train loss, Train accuracy, Test accuracy"
now = datetime.now()
np.savetxt(
    f"log/{now.strftime('%Y%m%d_%H%M%S')}.csv",
    result,
    fmt=["%d", "%f", "%f", "%f"],
    delimiter=",",
    header=header,
)
