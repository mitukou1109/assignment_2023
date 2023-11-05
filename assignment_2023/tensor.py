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
