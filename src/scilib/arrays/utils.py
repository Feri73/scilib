from types import ModuleType
from typing import Union, Tuple

import numpy as numpy_lib
import numpy.typing as npt

try:
    import scipy.stats as scipy_stats_lib
except:
    scipy_stats_lib: ModuleType = None

NPValue = npt.ArrayLike
NPIndex = Union[NPValue, slice]
NPAxis = Union[int, Tuple[int]]


def combine_axes(data: NPValue, axis: NPAxis, *, np=numpy_lib) -> npt.NDArray:
    if isinstance(axis, int):
        axis = (axis,)
    axis = tuple(axis)
    data = np.asarray(data)
    sh = tuple(a for a in range(data.ndim) if a not in axis)
    a = np.transpose(data, axis + sh)
    a = np.reshape(a, (numpy_lib.prod(a.shape[:len(axis)]), *a.shape[len(axis):]))
    return a


def ci(data: NPValue, axis: NPAxis, confidence: float = 0.95, keepdims: bool = False, *,
       np=numpy_lib, scipy_stats=scipy_stats_lib) -> npt.NDArray:
    a = combine_axes(data, axis, np=np)
    h = scipy_stats.sem(a) * scipy_stats.t.ppf((1 + confidence) / 2., len(a) - 1)
    if keepdims:
        h = np.expand_dims(h, axis)
    return h


def nanci(data: NPValue, axis: NPAxis, confidence: float = 0.95, keepdims: bool = False, *,
          np=numpy_lib, scipy_stats=scipy_stats_lib) -> npt.NDArray:
    a = combine_axes(data, axis, np=np)
    h = scipy_stats.sem(a, nan_policy='omit') * scipy_stats.t.ppf((1 + confidence) / 2.,
                                                                  (len(a) - np.sum(np.isnan(a), axis=0)) - 1)
    if isinstance(h, numpy_lib.ma.MaskedArray):
        h = np.where(np.logical_not(h.mask), h, np.nan)
    if keepdims:
        h = np.expand_dims(h, axis)
    return h


def nanttest(a: NPValue, b: NPValue, axis: NPAxis, keepdims: bool = False, *,
             np=numpy_lib, scipy_stats=scipy_stats_lib, **kwargs) -> npt.NDArray:
    a = combine_axes(a, axis, np=np)
    b = combine_axes(b, axis, np=np)
    h = scipy_stats.ttest_ind(a, b, nan_policy='omit', **kwargs).pvalue
    if isinstance(h, numpy_lib.ma.MaskedArray):
        h = np.where(np.logical_not(h.mask), h, np.nan)
    if keepdims:
        h = np.expand_dims(h, axis)
    return h


def non_nan_count(data: NPValue, axis: NPAxis, keepdims: bool = False, *, np=numpy_lib) -> npt.NDArray:
    res = np.count_nonzero(~np.isnan(data), axis=axis)
    if keepdims:
        res = np.expand_dims(res, axis)
    return res
