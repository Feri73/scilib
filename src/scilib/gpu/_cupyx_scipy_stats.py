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

    @staticmethod
    def sf(x, *args, **kwargs):
        x = cp.asnumpy(x)
        args = map(cp.asnumpy, args)
        kwargs = {k: cp.asnumpy(v) if isinstance(v, cp.ndarray) else v for k, v in kwargs.items()}
        return cp.asarray(scipy.stats.t.sf(x, *args, **kwargs))


class f:
    @staticmethod
    def sf(x, *args, **kwargs):
        x = cp.asnumpy(x)
        args = map(cp.asnumpy, args)
        kwargs = {k: cp.asnumpy(v) if isinstance(v, cp.ndarray) else v for k, v in kwargs.items()}
        return cp.asarray(scipy.stats.f.sf(x, *args, **kwargs))


class chi2:
    @staticmethod
    def ppf(q, *args, **kwargs):
        q = cp.asnumpy(q)
        args = map(cp.asnumpy, args)
        return cp.asarray(scipy.stats.chi2.ppf(q, *args, **kwargs))


class binom:
    @staticmethod
    def sf(*args, **kwargs):
        args = map(cp.asnumpy, args)
        return cp.asarray(scipy.stats.binom.sf(*args, **kwargs))

    @staticmethod
    def cdf(*args, **kwargs):
        args = map(cp.asnumpy, args)
        return cp.asarray(scipy.stats.binom.cdf(*args, **kwargs))


cstats.t = t
cstats.f = f
cstats.chi2 = chi2
cstats.binom = binom

sys.modules[__name__] = cstats
