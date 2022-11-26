from numbers import Number
from typing import Union, Iterable, Tuple

import numpy as np
import numpy.typing as npt

try:
    import scipy.stats
except:
    pass

NPValue = Union[npt.NDArray, Number, Iterable[Number]]
NPIndex = Union[NPValue, slice]
NPAxis = Union[int, Tuple[int]]


def ci(data: NPValue, axis: NPAxis, confidence: float = 0.95, keepdims: bool = False):
    if isinstance(axis, int):
        axis = (axis,)
    axis = tuple(axis)
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    sh = tuple(a for a in range(data.ndim) if a not in axis)
    a = np.transpose(data, axis + sh)
    a = np.reshape(a, (np.prod(a.shape[:len(axis)]), *a.shape[len(axis):]))
    h = scipy.stats.sem(a) * scipy.stats.t.ppf((1 + confidence) / 2., len(a) - 1)
    if keepdims:
        h = np.expand_dims(h, axis)
    return h


def nanci(data: NPValue, axis: NPAxis, confidence: float = 0.95, keepdims: bool = False):
    if isinstance(axis, int):
        axis = (axis,)
    axis = tuple(axis)
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    sh = tuple(a for a in range(data.ndim) if a not in axis)
    a = np.transpose(data, axis + sh)
    a = np.reshape(a, (np.prod(a.shape[:len(axis)]), *a.shape[len(axis):]))
    h = scipy.stats.sem(a, nan_policy='omit') * scipy.stats.t.ppf((1 + confidence) / 2.,
                                                                  (len(a) - np.sum(np.isnan(a), axis=0)) - 1)
    if isinstance(h, np.ma.MaskedArray):
        h = np.where(np.logical_not(h.mask), h, np.nan)
    if keepdims:
        h = np.expand_dims(h, axis)
    return h
