from typing import Callable, Union, Tuple

import dask.array as da
import dask
import sys
import numpy as np
import numpy.typing as npt

from ...utils import monkey_patch

for dtype in np.sctypeDict.values():
    if isinstance(dtype, type):
        setattr(da, dtype.__name__, dtype)


def _get_chunk_slice(shape, chunks: Union[int, Tuple[int, ...]], dtype=None):
    chunks = da.core.normalize_chunks(
        chunks=chunks, shape=shape, dtype=dtype
    )
    chunks_shape = tuple([len(c) for c in chunks])
    slices = np.empty(
        shape=chunks_shape + (len(chunks_shape), 2),
        dtype=int,
    )
    for ind in np.ndindex(chunks_shape):
        current_chunk = [chunk[i] for i, chunk in zip(ind, chunks)]
        starts = [int(np.sum(chunk[:i])) for i, chunk in zip(ind, chunks)]
        stops = [s + c for s, c in zip(starts, current_chunk)]
        slices[ind] = [[start, stop] for start, stop in zip(starts, stops)]

    return da.from_array(slices, chunks=(1,) * len(shape) + slices.shape[-2:]), chunks


def from_loader(chunks: Union[int, Tuple[int, ...]], load_chunk: callable, **load_kwargs) -> da.Array:
    dtype = load_kwargs['dtype']
    shape = load_kwargs['shape']

    if isinstance(chunks, int):
        chunks = (chunks, *[-1 for _ in range(len(shape) - 1)])

    if not isinstance(shape, tuple):
        shape = (shape,)

    chunked_slices, data_chunks = _get_chunk_slice(
        shape=shape,
        chunks=chunks,
        dtype=dtype,
    )
    num_dim = len(shape)

    def slice_chunk(slices, **kwargs):
        return load_chunk(_convert_slices(slices), **kwargs)

    data = da.map_blocks(
        slice_chunk,
        chunked_slices,
        **load_kwargs,
        dt=dtype,
        chunks=data_chunks,
        drop_axis=(
            num_dim,
            num_dim + 1,
        ),
    )
    return data


da.from_loader = from_loader


def _convert_slices(slices):
    return tuple([slice(s[0], s[1]) for s in np.squeeze(
        slices, axis=tuple([i for i in range(slices.ndim - 2) if slices.shape[i] == 1]))[()]])


def _slice_memmap(slices, **memap_kwargs):
    memmap_class = memap_kwargs.pop('memmap_class', np.memmap)
    dtype = memap_kwargs.pop('dt')
    data = memmap_class(**memap_kwargs, dtype=dtype)
    return data[slices]


def from_memmap(*, chunks: Union[int, Tuple[int, ...]], **memap_kwargs) -> da.Array:
    return from_loader(chunks=chunks, load_chunk=_slice_memmap, **memap_kwargs)


da.from_memmap = from_memmap


def from_delayed_array(a: npt.ArrayLike):
    def load(sl, **_):
        return a[sl]

    return from_loader(chunks=a.shape[0], load_chunk=load, shape=a.shape, dtype=a.dtype)


da.from_delayed_array = from_delayed_array


def from_any_array(array: np.ndarray, chunks: tuple[int, ...]) -> da.Array:
    if isinstance(array, np.memmap):
        return da.from_memmap(filename=array.filename,
                              shape=array.shape,
                              dtype=array.dtype,
                              mode='r',
                              chunks=chunks)
    else:
        return da.from_delayed_array(array).rechunk(chunks)


da.from_any_array = from_any_array


class set_chunks(monkey_patch):
    def __init__(self, original: Callable, chunks, *, rewrite_chunks: bool = None, rechunk: bool = False):
        self.chunks = chunks
        self.rewrite_chunks = rewrite_chunks
        self.rechunk = rechunk
        assert self.rechunk is False or self.rewrite_chunks is not True
        super().__init__(da, original, self.replaced)

    def replaced(self, *args, **kwargs):
        orig_chunks = kwargs.pop('chunks', self.chunks)
        chunks = self.chunks if self.rewrite_chunks is True or self.rewrite_chunks is None else orig_chunks
        if self.rechunk:
            return self.original(*args, **kwargs).rechunk(chunks)
        return self.original(*args, **kwargs, chunks=chunks)


da.set_chunks = set_chunks

sys.modules[__name__] = da
