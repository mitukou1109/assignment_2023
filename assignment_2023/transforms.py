import torch


def random_noise(x: torch.Tensor, prob: float):
    return torch.where(
        torch.randint(0, 100, size=x.shape) >= int(100 * prob), x, torch.rand(x.shape)
    )
