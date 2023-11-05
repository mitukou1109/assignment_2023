import torch.utils.data

from .tensor import Tensor


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 1,
        shuffle: bool = None,
        num_workers: int = 0,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def __iter__(self):
        i: list[torch.Tensor]
        for i in super().__iter__():
            yield (Tensor(i[0].detach().numpy().copy()), i[1].detach().numpy().copy())
