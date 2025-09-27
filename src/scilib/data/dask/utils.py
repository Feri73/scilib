import hashlib
import os
import warnings
from pathlib import Path
from typing import Union, Dict, Callable, Tuple

import dask
import numpy as np
from ..dask import array as da
from dask.diagnostics import ProgressBar

from ..memmap import AutoFlushMemmap
from ...path.path import UniversalPath
from ...gpu import xumpy as xp
from ...arrays import array as arr


def compute(*a, message, lock=False, config: dict = None, verbose=False):
    if config is None:
        config = {}
    if verbose:
        print('Compute:', message)
    with dask.config.set(config):
        if verbose:
            with ProgressBar():
                res = da.compute(*a, lock=lock)
        else:
            res = da.compute(*a, lock=lock)
    if verbose:
        print('Done')
    if len(a) == 1:
        return res[0]
    return res


def dx_store(path: str, message: str, *, flush_every: int = None,
             dx_config: dict = None, verbose=False, lock=False, **objs) -> Dict[str, AutoFlushMemmap]:
    if verbose:
        print('Storing:', message)

    path = UniversalPath(path)

    Path(path).mkdir(parents=True, exist_ok=True)

    uncertain_objs = {k: o for k, o in objs.items() if np.any([np.isnan(s) for s in o.shape])}
    if len(uncertain_objs) > 0:
        computed_objs = compute(*uncertain_objs.values(), message=f'Computing uncertain objects',
                                config=dx_config, verbose=verbose, lock=lock)
        objs = objs.copy()
        for k, o in zip(uncertain_objs.keys(), computed_objs):
            objs[k] = da.from_delayed_array(o)

    if flush_every is None:
        memmaps = tuple([AutoFlushMemmap(path / f'{obj_name}.npy', mode='w+', shape=obj.shape, dtype=obj.dtype)
                         for obj_name, obj in objs.items()])
    else:
        memmaps = tuple([AutoFlushMemmap(path / f'{obj_name}.npy',
                                         mode='w+', shape=obj.shape, dtype=obj.dtype, flush_every=flush_every)
                         for obj_name, obj in objs.items()])
    with dask.config.set(dx_config):
        if verbose:
            with ProgressBar():
                da.store(tuple([da.map_blocks(xp.asnumpy, obj) for obj in objs.values()]), memmaps, lock=lock)
        else:
            da.store(tuple([da.map_blocks(xp.asnumpy, obj) for obj in objs.values()]), memmaps, lock=lock)

    res = []
    for memmap in memmaps:
        memmap.flush()
        res.append(AutoFlushMemmap(UniversalPath(memmap.filename), mode='r', shape=memmap.shape, dtype=memmap.dtype))

    if verbose:
        print('Done')

    return {k: r for k, r in zip(objs.keys(), res)}


def spill_to_disk(*objs, message: str, chunkses: Tuple[Union[int, Tuple[int, ...]], ...], path: str,
                  flush_every: int = None, lock=False, dx_config: dict = None,
                  **kwargs) -> Union[Tuple[da.Array, ...], da.Array]:
    spills_dir = os.path.join(os.path.expandvars(path),
                              hashlib.sha256(message.encode()).hexdigest())
    res = {int(i): da.from_memmap(chunks=chunkses[int(i)], filename=UniversalPath(x.filename), dtype=x.dtype,
                                  shape=x.shape, memmap_class=AutoFlushMemmap)
           for i, x in dx_store(spills_dir, message, **kwargs, **{str(i): o for i, o in enumerate(objs)}, lock=lock,
                                flush_every=flush_every, dx_config=dx_config).items()}
    for i, o in enumerate(objs):
        if isinstance(o._meta, xp.ndarray):
            res[i] = as_dx(res[i])
    res = tuple([res[i] for i in range(len(res))])
    if len(res) == 1:
        return res[0]
    return res


def as_dx(a: Union[da.Array, arr.Array]) -> Union[da.Array, arr.Array]:
    if isinstance(a, da.Array):
        return da.map_blocks(xp.asarray, a)
    else:
        return a(as_dx(a.numpy))


def as_da(a: da.Array) -> da.Array:
    if isinstance(a, da.Array):
        return da.map_blocks(xp.asnumpy, a)
    else:
        return a(as_da(a.numpy))


def get_chunks(a: arr.Array, chunks: Dict[str, Union[int, Tuple[int, ...]]], unspecified_policy: str = '-1') \
        -> Tuple[Union[int, Tuple[int, ...]], ...]:
    unspecified_policy == -1 or unspecified_policy == 'keep'
    res = [-1 if unspecified_policy == '-1' else a.numpy.chunks[i] for i in range(a.ndim)]
    for view_name in chunks:
        res[getattr(a, view_name).view.axis] = chunks[view_name]
    return tuple(res)


def rechunk(a: arr.Array, chunks: Dict[str, Union[int, Tuple[int, ...]]], unspecified_policy: str = '-1') -> arr.Array:
    new_chunks = get_chunks(a, chunks, unspecified_policy)
    if all([new == -1 and len(old) == 1 or old == new for old, new in zip(a.numpy.chunks, new_chunks)]):
        return a
    return a(a.numpy.rechunk(new_chunks))


def chunks_as_dict(a: arr.Array) -> Dict[str, Union[int, Tuple[int, ...]]]:
    return {view_name: a.numpy.chunks[getattr(a, view_name).view.axis] for view_name in a.views}


def get_dask_func(func: Callable, **dask_kwargs) -> Callable:
    def f(a, **kwargs):
        if 'axis' in kwargs and len(a.chunks[kwargs['axis']]) != 1:
            warnings.warn(f'Axis {kwargs["axis"]} is not chunked!', RuntimeWarning)
        return da.map_blocks(func, a,
                             dtype=a.dtype,
                             meta=xp.array((), dtype=a.dtype),
                             **dask_kwargs,
                             **kwargs)

    return f


def get_dask_reduce_func(func: Callable, *, pos_kws: tuple[str, ...] = None, n_out: int = None,
                         **dask_kwargs) -> Callable:
    if pos_kws is None:
        pos_kws = ()

    _np = dask_kwargs.get('np', np)

    def f(a, *, axis, keepdims, **kwargs):
        if n_out is None:
            _func = func
        else:
            _func = lambda *a, **kwa: _np.stack(func(*a, **kwa), axis=0)
        axes = axis if isinstance(axis, tuple) else [axis]

        all_dask_inps = [a] + [kwargs.pop(kw) for kw in pos_kws]
        for d in all_dask_inps[1:]:
            assert a.ndim == d.ndim
        chunks = []
        for i in range(a.ndim):
            for d in all_dask_inps:
                if d.shape[i] != 1:
                    chunks.append(d.chunks[i])
                    break
            else:
                chunks.append((1,))
        for i in range(a.ndim):
            assert all(len(d.chunks[i]) == 1 for d in all_dask_inps) or \
                   all(d.shape[i] == 1 or d.chunks[i] == chunks[i] for d in all_dask_inps)

        res = da.map_blocks(_func, *all_dask_inps,
                            drop_axis=axis,
                            dtype=a.dtype,
                            meta=_np.array((), dtype=a.dtype),
                            **dask_kwargs,
                            axis=axis,
                            keepdims=keepdims,
                            new_axis=None if n_out is None else 0,
                            chunks=None if n_out is None else
                            ((n_out,),) + tuple(a.chunks[i] for i in range(a.ndim) if i not in axes),
                            **kwargs)
        if n_out is not None:
            assert res.shape[0] == n_out
            if keepdims:
                return tuple(da.expand_dims(res[i], axis=axis) for i in range(n_out))
            else:
                return tuple(res[i] for i in range(n_out))
        if keepdims:
            res = da.expand_dims(res, axis=axis)
        return res

    return f
