from types import ModuleType
from typing import Union, Tuple, Literal

import numpy as numpy_lib
import numpy.typing as npt

from ..arrays.utils import combine_axes, non_nan_count, uncombine_axes

try:
    import scipy.stats as scipy_stats_lib
    import scipy.special as scipy_special_lib
except:
    scipy_stats_lib: ModuleType = None
    scipy_special_lib: ModuleType = None

NPValue = npt.ArrayLike
NPIndex = Union[NPValue, slice]
NPAxis = Union[int, Tuple[int]]


def nanci_garwood(counts: NPValue, axis: NPAxis, confidence: float = 0.95, keepdims: bool = False, *,
                  np=numpy_lib, scipy_stats=scipy_stats_lib) -> Tuple[npt.NDArray, npt.NDArray]:
    counts = combine_axes(counts, axis, np=np)
    K = np.nansum(counts, axis=0)
    T = non_nan_count(counts, axis=0, np=np)
    alpha = 1. - confidence
    mu_lower = np.where(K == 0, 0., .5 * scipy_stats.chi2.ppf(alpha / 2., 2 * K)) / T
    mu_upper = (0.5 * scipy_stats.chi2.ppf(1. - alpha / 2., 2 * (K + 1))) / T
    if keepdims:
        mu_lower = np.expand_dims(mu_lower, axis)
        mu_upper = np.expand_dims(mu_upper, axis)
    return mu_lower, mu_upper


def _binom_logpmf(x, n, p, *, np=numpy_lib, scipy_special=scipy_special_lib):
    logC = scipy_special.gammaln(n + 1) - scipy_special.gammaln(x + 1) - scipy_special.gammaln(n - x + 1)
    term1 = np.where(x == 0, 0.0, x * np.log(p))
    term2 = np.where((n - x) == 0, 0.0, (n - x) * np.log1p(-p))
    return logC + term1 + term2


def _binomtest_two_sided(c, n, p, *, np=numpy_lib, scipy_special=scipy_special_lib):
    shape = np.broadcast_shapes(c.shape, n.shape, p.shape)
    n_max = int(n.max())
    xs = np.arange(n_max + 1, dtype=np.int64).reshape((1,) * len(shape) + (n_max + 1,))
    n_exp = np.expand_dims(n, axis=-1)
    p_exp = np.expand_dims(p, axis=-1)

    mask_x = xs <= n_exp
    logpmf = _binom_logpmf(xs.astype(np.float64), n_exp.astype(np.float64), p_exp, np=np, scipy_special=scipy_special)
    logpmf = np.where(mask_x, logpmf, -np.inf)
    pmf = np.exp(logpmf, dtype=np.float64)

    c_idx = np.expand_dims(c, axis=-1)
    pmf_c = np.take_along_axis(pmf, c_idx, axis=-1)

    tol = 1e-15
    le_mask = (pmf <= (pmf_c + tol)) & mask_x
    pvals = (pmf * le_mask).sum(axis=-1, dtype=np.float64)

    return pvals


def _binomtest_one_sided(c, n, p, *, scipy_stats=scipy_stats_lib, alternative='greater'):
    if alternative == 'greater':
        vals = scipy_stats.binom.sf(c - 1, n, p)
    elif alternative == 'less':
        vals = scipy_stats.binom.cdf(c, n, p)
    return vals


def nan_poisson_exact_test_summary_statistics(
        total_count1: NPValue, total_count2: NPValue,
        exposure1: NPValue, exposure2: NPValue,
        alternative: Union[Literal['two-sided'], Literal['greater'], Literal['less']], *,
        np=numpy_lib, scipy_special=scipy_special_lib,
        scipy_stats=scipy_stats_lib) -> npt.NDArray:
    if alternative == 'two-sided':
        return _binomtest_two_sided(total_count1, total_count1 + total_count2, exposure1 / (exposure1 + exposure2),
                                    np=np, scipy_special=scipy_special)
    else:
        return _binomtest_one_sided(total_count1, total_count1 + total_count2, exposure1 / (exposure1 + exposure2),
                                    alternative=alternative, scipy_stats=scipy_stats)


def nan_poisson_exact_test(counts1: NPValue, counts2: NPValue,
                           exposures1: Union[NPValue, float], exposures2: Union[NPValue, float],
                           axis: NPAxis, keepdims: bool = False, *,
                           np=numpy_lib, scipy_special=scipy_special_lib, scipy_stats=scipy_stats_lib,
                           **kwargs) -> npt.NDArray:
    counts1 = combine_axes(counts1, axis, np=np)
    counts2 = combine_axes(counts2, axis, np=np)
    if not isinstance(exposures1, float):
        exposures1 = combine_axes(exposures1, axis, np=np)
    if not isinstance(exposures2, float):
        exposures2 = combine_axes(exposures2, axis, np=np)
    total_count1 = np.nansum(counts1, axis=0)
    total_count2 = np.nansum(counts2, axis=0)
    exposures1 = np.nansum(np.where(np.isnan(counts1), np.nan, 1) * exposures1, axis=0)
    exposures2 = np.nansum(np.where(np.isnan(counts2), np.nan, 1) * exposures2, axis=0)
    h = nan_poisson_exact_test_summary_statistics(total_count1, total_count2,
                                                  exposures1, exposures2,
                                                  np=np, scipy_special=scipy_special, scipy_stats=scipy_stats, **kwargs)
    if keepdims:
        h = np.expand_dims(h, axis)
    return h


def nan_welch_test(a: NPValue, b: NPValue, axis: NPAxis,
                   keepdims: bool = False, *,
                   np=numpy_lib, scipy_stats=scipy_stats_lib) -> npt.NDArray:
    a = combine_axes(a, axis, np=np)
    a_mean = np.nanmean(a, axis=0)
    a_std = np.nanstd(a, axis=0)
    a_count = non_nan_count(a, axis=0, np=np)
    b = combine_axes(b, axis, np=np)
    b_mean = np.nanmean(b, axis=0)
    b_std = np.nanstd(b, axis=0)
    b_count = non_nan_count(b, axis=0, np=np)
    t_stat = (a_mean - b_mean) / np.sqrt(a_std ** 2 / a_count + b_std ** 2 / b_count)
    df_num = (a_std ** 2 / a_count + b_std ** 2 / b_count) ** 2
    df_den = ((a_std ** 2 / a_count) ** 2 / (a_count - 1)) + ((b_std ** 2 / b_count) ** 2 / (b_count - 1))
    df = df_num / df_den
    p_value = 2 * (1 - scipy_stats.t.cdf(np.abs(t_stat), df))
    if keepdims:
        p_value = np.expand_dims(p_value, axis)
    return p_value


def _benjamini_hochberg(p: NPValue, axis=0, *, np=numpy_lib) -> NPValue:
    p = np.swapaxes(p, axis, -1)
    shape = p.shape
    q = np.empty_like(p)
    for idx in numpy_lib.ndindex(shape[:-1]):
        q[idx] = _bh_1d(p[idx], np=np)
    q = np.swapaxes(q, -1, axis)
    return q


def _bh_1d(p_1d: NPValue, *, np=numpy_lib) -> NPValue:
    q = np.full_like(p_1d, numpy_lib.nan, dtype=np.float64)

    valid = ~np.isnan(p_1d)
    m = int(valid.sum())
    if m == 0:
        return q

    order = np.argsort(p_1d[valid])
    p_sorted = p_1d[valid][order]

    ranks = np.arange(1, m + 1, dtype=np.float64)
    q_raw = (m / ranks) * p_sorted

    q_monotone = _cummin_from_right(q_raw, axis=0, np=np)
    q_monotone = np.clip(q_monotone, 0.0, 1.0)

    q_vals = np.empty(m, dtype=np.float64)
    q_vals[order] = q_monotone
    q[valid] = q_vals

    return q


def _cummin_from_right(a, axis, *, np=numpy_lib):
    if hasattr(np.minimum, 'accumulate'):
        try:
            a_flip = np.flip(a, axis=axis)
            cm_flip = np.minimum.accumulate(a_flip, axis=axis)
            return np.flip(cm_flip, axis=axis)
        except NotImplementedError:
            pass

    a_move = np.moveaxis(a, axis, -1)
    out = a_move.copy()

    for i in range(a_move.shape[-1] - 2, -1, -1):
        out[..., i] = np.minimum(out[..., i], out[..., i + 1])
    return np.moveaxis(out, -1, axis)


def nan_bh_correction(p: NPValue, axis: NPAxis, *, np=numpy_lib) -> npt.NDArray:
    return uncombine_axes(_benjamini_hochberg(combine_axes(p, axis, np=np), np=np), axis, np.asarray(p).shape, np=np)
