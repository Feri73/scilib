from typing import Union

from ..arrays.array import Array, ArrayView
from . import xumpy as xp
import numpy as np
import numpy.typing as npt

from ..utils import Config


def to_xp(array: Union[Array, ArrayView, npt.ArrayLike]) -> Union[Array, ArrayView, npt.ArrayLike]:
    if isinstance(array, Array) or isinstance(array, ArrayView):
        return array.copy(np=xp)(xp.asarray(array.numpy))
    return xp.asarray(array)


def to_numpy(array: Union[Array, ArrayView, npt.ArrayLike]) -> Union[Array, ArrayView, npt.ArrayLike]:
    if isinstance(array, Array) or isinstance(array, ArrayView):
        return array.copy(np=np)(xp.asnumpy(array.numpy))
    return xp.asnumpy(array)

class use_xp:
    def __init__(self, **inputs):
        self.vars = Config(**inputs)
        self.outputs = Config()

    def __enter__(self):
        try:
            xp.get_default_memory_pool().free_all_blocks()
        except AttributeError:
            pass
        for k in self.vars:
            self.vars[k] = to_xp(self.vars[k])
        return self.vars, self.outputs

    def __exit__(self, exc_type, exc_value, traceback):
        for k in self.outputs:
            self.outputs[k] = to_numpy(self.outputs[k])
        self.vars.clear()
        try:
            xp.get_default_memory_pool().free_all_blocks()
        except AttributeError:
            pass