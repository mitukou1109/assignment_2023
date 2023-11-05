from .tensor import Tensor


class SGD:
    def __init__(self, params: Tensor, lr: float, momentum: float = 0):
        self.params = params
        self.lr = lr
        self.momentum = momentum

    def zero_grad(self):
        pass

    def step(self, grad: Tensor):
        w = self.params
        while grad is not None:
            w -= self.lr * grad
            grad = grad.prev
            w: Tensor = w.prev
