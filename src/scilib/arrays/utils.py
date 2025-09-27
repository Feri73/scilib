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
    """
    Combine multiple axes of an array into a single axis.
    The order of combination is the same as the order of the axes in the input.
    """
    if isinstance(axis, int):
        axis = (axis,)
    axis = tuple(axis)
    data = np.asarray(data)
    sh = tuple(a for a in range(data.ndim) if a not in axis)
    a = np.transpose(data, axis + sh)
    a = np.reshape(a, (numpy_lib.prod(a.shape[:len(axis)]), *a.shape[len(axis):]))
    return a


def uncombine_axes(data: npt.NDArray, axis: NPAxis, original_shape: Tuple[int, ...], *, np=numpy_lib) -> npt.NDArray:
    if isinstance(axis, int):
        axis = (axis,)
    axis = tuple(axis)
    data = np.asarray(data)
    axis_shape = tuple(original_shape[a] for a in axis)
    sh = tuple(original_shape[a] for a in range(len(original_shape)) if a not in axis)
    a = np.reshape(data, axis_shape + sh)
    inv_perm = tuple(np.argsort(np.array(axis + tuple(i for i in range(len(original_shape)) if i not in axis))
                                ).tolist())
    a = np.transpose(a, inv_perm)
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


def nan_ndargmax(data: NPValue, axis: NPAxis, keepdims: bool = False, return_values: bool = False,
                 *, np=numpy_lib) -> Tuple[npt.NDArray, ...]:
    a = combine_axes(data, axis, np=np)
    flat_inds = np.nanargmax(a, axis=0)
    unflat_inds = np.unravel_index(flat_inds,
                                   tuple(data.shape[i] for i in ((axis,) if isinstance(axis, int) else axis)))
    if keepdims:
        unflat_inds = tuple(np.expand_dims(i, axis) for i in unflat_inds)
    if not return_values:
        return unflat_inds

    vals = np.take_along_axis(a, flat_inds[None], axis=0)[0]
    if keepdims:
        vals = np.expand_dims(vals, axis)
    return *unflat_inds, vals
