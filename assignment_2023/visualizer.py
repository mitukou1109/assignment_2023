import glob
import sys

import matplotlib.pyplot as plt
import numpy as np

csv_path = max(glob.glob("log/*.csv")) if len(sys.argv) < 2 else sys.argv[1]
result = np.loadtxt(csv_path, delimiter=",")

if len(sys.argv) >= 3:
    result = result[result[:, 0] <= int(sys.argv[2])]

epochs = result[:, 0]
train_loss = result[:, 1]
train_acc = result[:, 2]
test_acc = result[:, 3]

if epochs[-1] < 20:
    xticks = np.arange(1, epochs[-1] + 1)
else:
    step = 10 if epochs[-1] <= 100 else 50
    xticks = np.concatenate(
        [
            np.array([1]),
            np.arange(step, (epochs[-1] // step + 1) * step + 1, step),
        ]
    )

loss_curve = plt.figure(num="Loss")
loss_curve.subplots_adjust(left=0.0875, right=0.975, bottom=0.1, top=0.975)
ax = loss_curve.add_subplot()
ax.plot(epochs, train_loss)
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_xticks(xticks)
ax.set_xlim(1, epochs[-1])
ax.grid(axis="y")

learning_curve = plt.figure(num="Learning curve")
ax = learning_curve.add_subplot()
ax.plot(epochs, train_acc, label="train")
ax.plot(epochs, test_acc, label="test")
ax.legend(loc="lower right")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
ax.set_xticks(xticks)
ax.set_xlim(1, epochs[-1])
y_min = np.round(min(np.min(train_acc), np.min(test_acc)), 1)
y_step = 0.025 if 1 - y_min <= 0.2 else 0.05 if 1 - y_min <= 0.5 else 0.1
ax.set_yticks(np.arange(y_min, 1.01, y_step))
ax.set_ylim(y_min - (1 - y_min) * 0.05, 1 + (1 - y_min) * 0.05)
ax.grid(axis="y")
learning_curve.subplots_adjust(
    left=0.12 - y_step * 0.4, right=0.975, bottom=0.1, top=0.975
)

plt.show()
