from functools import partial

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from . import net as nn
from . import transforms

import sys

batch_size = 512
features = [28 * 28, 1024, 512, 10]
learning_rate = 0.5
epochs = 100
noise_prob = 0
seed = 13

show_data_sample = False
show_optimal_stimuli = True

initial_params: nn.Tensor = None
result = np.ndarray((epochs, 4))
result[:, 0] = np.arange(epochs) + 1
starting_epoch = 0

if len(sys.argv) >= 2:
    checkpoint: dict[str, np.ndarray] = np.load(sys.argv[1])
    batch_size = int(checkpoint["batch_size"][0])
    features[1:-1] = checkpoint["hidden_layers"]
    learning_rate = float(checkpoint["learning_rate"][0])
    noise_prob = float(checkpoint["noise_prob"][0])
    seed = int(checkpoint["seed"][0])

    param_keys = sorted(k for k, _ in checkpoint.items() if k.startswith("w_"))
    initial_params = nn.Tensor(checkpoint[param_keys[0]])
    p = initial_params
    for param in param_keys[1:]:
        p.prev = nn.Tensor(checkpoint[param])
        p = p.prev

    starting_epoch = checkpoint["result"].shape[0]
    assert epochs > starting_epoch
    result[:starting_epoch] = checkpoint["result"]

    print(
        f"Loaded checkpoint: batch_size = {batch_size}, hidden_layers = {features[1:-1]}, learning_rate = {learning_rate}, noise_prob = {noise_prob}, seed = {seed}"
    )

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

if initial_params is not None:
    net.set_params(initial_params)

now = datetime.now()
checkpoint_file = f"checkpoint/{now.strftime('%Y%m%d_%H%M%S')}.npz"
log_file = f"log/{now.strftime('%Y%m%d_%H%M%S')}.csv"
header = f"Batch size: {batch_size}, Hidden layers: {features[1:-1]}, Learning rate: {learning_rate}, Noise: {int(noise_prob * 100)}%"
header += "\nEpoch, Train loss, Train accuracy, Test accuracy"

train_loss = result[:, 1]
train_acc = result[:, 2]
test_acc = result[:, 3]

for i in np.arange(starting_epoch, epochs):
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
    optimal_stimuli: 
    for x, t in test_loader:
        x = x.reshape(-1, features[0])
        y = net(x)
        acc += net.calc_acc(y, t)
        while y is not None:
            optimal_stimuli.append((y.max(axis=0), x[y.argmax(axis=0)]))

    test_acc[i] = acc / len(test_loader)

    print(
        f"train loss: {train_loss[i]}, train accuracy: {train_acc[i]}, test accuracy: {test_acc[i]}",
        end="\n\n",
    )

    np.savez_compressed(
        checkpoint_file,
        batch_size=np.array([batch_size]),
        hidden_layers=np.array(features[1:-1]),
        learning_rate=np.array([learning_rate]),
        noise_prob=np.array([noise_prob]),
        seed=np.array([seed]),
        **net.parameters().expand("w"),
        result=result[: i + 1],
    )

    np.savetxt(
        log_file,
        result[: i + 1],
        fmt=["%d", "%f", "%f", "%f"],
        delimiter=",",
        header=header,
    )
