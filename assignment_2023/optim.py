from .tensor import Tensor


class SGD:
    def __init__(self, params: Tensor, lr: float):
        self.params = params
        self.lr = lr

    def step(self, grad: Tensor):
        w: Tensor = self.params
        while grad is not None:
            w -= self.lr * grad
            grad = grad.prev
            w = w.prev
