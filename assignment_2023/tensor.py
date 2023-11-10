from typing import Any, Self

import numpy as np


class Tensor(np.ndarray):
    def __new__(cls, data: Any = None, prev: Self = None) -> None:
        obj = np.asarray(data).view(cls)
        obj.prev = prev
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self.prev = getattr(obj, "prev", None)

    def expand(self, prefix: str) -> dict[str, np.ndarray]:
        expanded_data: dict[str, np.ndarray] = {}
        data = self
        count = 0
        while data is not None:
            expanded_data[f"{prefix}_{count}"] = data
            data = data.prev
            count += 1
        return expanded_data
