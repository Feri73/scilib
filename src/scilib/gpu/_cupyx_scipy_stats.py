import sys

import cupy as cp
import numpy as np

import cupyx.scipy.stats as cstats
import scipy.stats


def sem(a: cp.ndarray, axis=0, ddof=1, nan_policy='propagate') -> cp.ndarray:
    if nan_policy == 'propagate':
        n = int(np.prod(np.asarray(a.shape)[np.asarray(axis)]))
        return cp.std(a, axis=axis, ddof=ddof) / n ** 0.5
    elif nan_policy == 'raise':
        if cp.any(cp.isnan(a)):
            raise ValueError('NaNs detected')
    n = cp.sum(cp.logical_not(cp.isnan(a)).astype(cp.int32), axis=axis)
    return cp.nanstd(a, axis=axis, ddof=ddof) / n ** 0.5


cstats.sem = sem


class t:
    @staticmethod
    def ppf(q, *args, **kwargs):
        q = cp.asnumpy(q)
        args = map(cp.asnumpy, args)
        return cp.asarray(scipy.stats.t.ppf(q, *args, **kwargs))

    @staticmethod
    def cdf(x, *args, **kwargs):
        x = cp.asnumpy(x)
        args = map(cp.asnumpy, args)
        return cp.asarray(scipy.stats.t.cdf(x, *args, **kwargs))


cstats.t = t

sys.modules[__name__] = cstats
