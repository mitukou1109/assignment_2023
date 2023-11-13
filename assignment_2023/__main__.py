import os
import sys
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
hidden_layer_features = [256, 64]
learning_rate = 0.5
noise_prob = 0
seed = 13

train = True
show_data_sample = False
show_optimal_stimuli = False
show_receptive_field = False

initial_params: nn.Tensor = None
result = np.ndarray((epochs, 4))
result[:, 0] = np.arange(epochs) + 1
starting_epoch = 0 if train else epochs - 1

file_basename = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_file = f"checkpoint/{file_basename}.npz"

if len(sys.argv) >= 2:
    checkpoint_file = sys.argv[1]
    file_basename = os.path.splitext(os.path.basename(checkpoint_file))[0]

    checkpoint: dict[str, np.ndarray]
    with np.load(checkpoint_file) as checkpoint:
        batch_size = int(checkpoint["batch_size"][0])
        hidden_layer_features = checkpoint["hidden_layer_features"].tolist()
        learning_rate = float(checkpoint["learning_rate"][0])
        noise_prob = float(checkpoint["noise_prob"][0])
        seed = int(checkpoint["seed"][0])

        param_keys = sorted(k for k, _ in checkpoint.items() if k.startswith("w_"))
        initial_params = nn.Tensor(checkpoint[param_keys[0]])
        p = initial_params
        for param in param_keys[1:]:
            p.prev = nn.Tensor(checkpoint[param])
            p = p.prev

        if train:
            starting_epoch = checkpoint["result"].shape[0]
            assert epochs > starting_epoch
            result[:starting_epoch] = checkpoint["result"]

    print(
        f"Loaded checkpoint: batch_size = {batch_size}, hidden_layer_features = {hidden_layer_features}, learning_rate = {learning_rate}, noise_prob = {noise_prob}, seed = {seed}"
    )

log_header = f"Batch size: {batch_size}, Hidden layer features: {hidden_layer_features}, Learning rate: {learning_rate}, Noise: {int(noise_prob * 100)}%"
log_header += "\nEpoch, Train loss, Train accuracy, Test accuracy"

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
    plt.imshow(sample, cmap="gray_r")
    plt.axis("off")
    plt.show()

input_rows = 28
input_cols = 28
output_features = 10

net = nn.Net([input_rows * input_cols] + hidden_layer_features + [output_features])
optimizer = nn.SGD(net.parameters(), learning_rate)
criterion = nn.CrossEntropyLoss(net.parameters())

if initial_params is not None:
    net.set_params(initial_params)

train_loss = result[:, 1]
train_acc = result[:, 2]
test_acc = result[:, 3]

hidden_layer_output = np.zeros((hidden_layer_features[0],))
optimal_stimuli = np.zeros((hidden_layer_features[0], input_rows, input_cols))

for i in np.arange(starting_epoch, epochs):
    if train:
        print(f"epoch {i + 1}")

        acc = 0
        loss = 0
        for x, t in train_loader:
            x = x.reshape(-1, input_rows * input_cols)
            y = net(x)
            acc += net.calc_acc(y, t)
            loss += criterion(y, t)
            grad = criterion.backward()
            optimizer.step(grad)

        train_acc[i] = acc / len(train_loader)
        train_loss[i] = loss / len(train_loader)

    acc = 0
    hidden_layer_output.fill(0)
    optimal_stimuli.fill(0)
    for x, t in test_loader:
        x_flat = x.reshape(-1, input_rows * input_cols)
        y = net(x_flat)
        acc += net.calc_acc(y, t)

        if show_optimal_stimuli and (i == epochs - 1 or not train):
            y: nn.Tensor = net.layers["fc1"](x_flat)
            optimal_stimuli = np.where(
                hidden_layer_output < y.max(axis=0),
                x[y.argmax(axis=0)].squeeze().T,
                optimal_stimuli.T,
            ).T
            hidden_layer_output = np.fmax(hidden_layer_output, y.max(axis=0))

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
        os.makedirs("checkpoint", exist_ok=True)
        np.savez_compressed(
            checkpoint_file,
            batch_size=np.array([batch_size]),
            hidden_layer_features=np.array(hidden_layer_features),
            learning_rate=np.array([learning_rate]),
            noise_prob=np.array([noise_prob]),
            seed=np.array([seed]),
            **net.parameters().expand("w"),
            result=result[: i + 1],
        )

        os.makedirs("log", exist_ok=True)
        np.savetxt(
            f"log/{file_basename}.csv",
            result[: i + 1],
            fmt=["%d", "%f", "%f", "%f"],
            delimiter=",",
            header=log_header,
        )

if show_optimal_stimuli:
    max_rows = 32
    if (
        optimal_stimuli.shape[0] > max_rows
        and (rem := optimal_stimuli.shape[0] % max_rows) != 0
    ):
        optimal_stimuli = np.vstack(
            [optimal_stimuli, np.zeros((max_rows - rem, input_rows, input_cols))]
        )

    rows = max(1, optimal_stimuli.shape[0] // max_rows)
    cols = min(optimal_stimuli.shape[0], max_rows)
    optimal_stimuli = (
        optimal_stimuli.reshape(rows, cols, input_rows, input_cols)
        .swapaxes(1, 2)
        .reshape(rows * input_rows, cols * input_cols)
    )
    fig = plt.figure(num="Optimal stimuli")
    plt.imshow(optimal_stimuli, cmap="gray_r")
    plt.axis("off")
    os.makedirs("log", exist_ok=True)
    fig.savefig(f"log/{file_basename}_optimal_stimuli.png", bbox_inches="tight")

cmap_list = [
    "Greys",
    "Purples",
    "Blues",
    "Greens",
    "Oranges",
    "Reds",
    "YlOrBr",
    "YlOrRd",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",
]
if show_receptive_field and hidden_layer_features[0] <= len(cmap_list):
    fig = plt.figure(num="Receptive field")
    weight = net.layers["fc1"].w[:, : input_rows * input_cols].clip(min=0.5)
    w: np.ndarray
    for i, w in zip(np.random.permutation(len(cmap_list)), weight):
        plt.imshow(w.reshape(input_rows, input_cols), cmap=cmap_list[i], alpha=0.5)
    plt.axis("off")
    os.makedirs("log", exist_ok=True)
    fig.savefig(f"log/{file_basename}_receptive_field.png", bbox_inches="tight")

plt.show()
