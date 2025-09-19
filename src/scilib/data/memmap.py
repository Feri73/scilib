import os
import threading
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

from ..path.path import UniversalPath
from ..utils import Config as C, load_pickle, dump_pickle

Index = Union[int, Tuple[int, ...]]


class DiskObj:
    def __init__(self, path: str, pickle, default=None):
        self.__path = path
        self.__pickle = pickle

        self.__obj = default
        if os.path.exists(path):
            self.__obj = load_pickle(pickle, path)

    @property
    def obj(self):
        return self.__obj

    @obj.setter
    def obj(self, obj):
        self.__obj = obj

    def flush(self) -> None:
        Path(self.__path).parent.absolute().mkdir(parents=True, exist_ok=True)
        dump_pickle(self.__pickle, self.__path, self.__obj)

    def __enter__(self) -> 'DiskObj':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.flush()


class NumpyMemmap:
    def __init__(self, path: str, max_indexing_shape: Index = None, example_point=None):
        max_indexing_shape = self.__index_to_tuple(max_indexing_shape)

        self.__path = path

        import pickle
        self.__metadata = DiskObj(f'{self.__path}/metadata.pckl', pickle)

        if self.__metadata.obj is None:
            self.__metadata.obj = C(max_indexing_shape=max_indexing_shape, dtype=None, data_shape=None)
        if example_point is not None:
            self.__use_example(example_point)
        assert self.__metadata.obj.max_indexing_shape is not None and \
               (max_indexing_shape == self.__metadata.obj.max_indexing_shape or max_indexing_shape is None)

        self.__opened_file = None
        self.__opened_filename = None

        if os.path.isdir(self.__path):
            all_files = list(sorted(filter(lambda x: x.endswith('.npy'), os.listdir(self.__path))))
            if len(all_files) == 1:
                self.__open(all_files[0][:-4], False)

    def flush(self) -> None:
        if self.__opened_file is not None:
            self.__opened_file.flush()
        self.__metadata.flush()

    def __use_example(self, example_point) -> None:
        if not isinstance(example_point, np.ndarray):
            example_point = np.array(example_point)
        if self.__metadata.obj.dtype is None:
            self.__metadata.obj.dtype = example_point.dtype
        else:
            assert self.__metadata.obj.data_shape == example_point.shape
        if self.__metadata.obj.data_shape is None:
            self.__metadata.obj.data_shape = example_point.shape
        else:
            assert self.__metadata.obj.data_shape == example_point.shape

    def numpy(self, inds_inside_numpy: Index) -> npt.NDArray:
        _ = self[inds_inside_numpy]
        return self.__opened_file

    def __open(self, filename: str, flush: bool) -> None:
        if flush:
            self.flush()
        del self.__opened_file
        if os.path.isfile(f'{self.__path}/{filename}.npy'):
            mode = 'r+'
        else:
            mode = 'w+'
        self.__opened_file = np.memmap(f'{self.__path}/{filename}.npy', mode=mode, dtype=self.__metadata.obj.dtype,
                                       shape=(*self.__metadata.obj.max_indexing_shape, *self.__metadata.obj.data_shape))
        self.__opened_filename = filename

    @staticmethod
    def __index_to_tuple(ind: Index) -> Tuple[int, ...]:
        if isinstance(ind, int):
            ind = (ind,)
        return ind

    def __inds2open(self, inds: Tuple[int, ...]) -> Tuple[int, ...]:
        parts = list(
            dim_ind // dim_max_len for dim_max_len, dim_ind in zip(self.__metadata.obj.max_indexing_shape, inds))
        filename = '_'.join(map(str, parts))
        if self.__opened_filename != filename:
            self.__open(filename, True)
        return tuple(inds - part * self.__metadata.obj.max_indexing_shape[dim]
                     for dim, (inds, part) in enumerate(zip(inds, parts)))

    def __getitem__(self, inds: Index) -> npt.NDArray:
        inds = self.__inds2open(self.__index_to_tuple(inds))
        return self.__opened_file[inds]

    def __setitem__(self, inds: Index, value):
        if self.__metadata.obj.dtype is None or self.__metadata.obj.data_shape is None:
            self.__use_example(value)
        inds = self.__inds2open(self.__index_to_tuple(inds))
        self.__opened_file[inds] = value

    def close(self) -> None:
        self.flush()
        if self.__opened_file is not None:
            del self.__opened_file
            self.__opened_file = None
            self.__opened_filename = None

    def __enter__(self) -> 'NumpyMemmap':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def merge_all(self, path: str) -> 'NumpyMemmap':
        from tqdm import tqdm

        all_files = list(filter(lambda x: x.endswith('.npy'), os.listdir(self.__path)))
        all_parts = np.array(list(map(lambda x: list(map(int, x[:-4].split('_'))), all_files)))
        max_parts = np.max(all_parts, axis=0)
        res = NumpyMemmap(path, tuple((max_part + 1) * dim_shape for max_part, dim_shape in
                                      zip(max_parts, self.__metadata.obj.max_indexing_shape)),
                          np.zeros(self.__metadata.obj.data_shape, dtype=self.__metadata.obj.dtype))
        _ = res[tuple([0] * len(all_parts))]
        for part, file in tqdm(list(zip(all_parts, all_files))):
            inds = tuple(slice(dim_part * dim_shape, (dim_part + 1) * dim_shape)
                         for dim_part, dim_shape in zip(part, self.__metadata.obj.max_indexing_shape))
            self.__open(file[:-4], True)
            res.__opened_file[inds] = self.__opened_file
        res.flush()
        return res


class SerializableMemmap(np.memmap):
    def __new__(cls, filename, dtype=None, mode='r+', offset=0, shape=None, order='C'):
        obj = np.memmap.__new__(cls, filename, dtype=dtype, mode=mode, offset=offset, shape=shape, order=order)
        obj._sp_filename = filename
        obj._sp_mode = mode
        obj._sp_offset = offset
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._sp_filename = getattr(obj, '_sp_filename', None)
        self._sp_mode = getattr(obj, '_sp_mode', 'r+')
        self._sp_offset = getattr(obj, '_sp_offset', 0)

    def __reduce__(self):
        self.flush()
        return self.__class__, (self._sp_filename, self.dtype, self._sp_mode, int(self._sp_offset), self.shape)


class AutoFlushMemmap(SerializableMemmap):
    def __new__(cls, filename, *args, flush_every: int = -1, **kwargs):
        if not isinstance(filename, UniversalPath):
            filename = UniversalPath(filename)
        obj = super().__new__(cls, filename, *args, **kwargs)
        obj._flush_every = flush_every
        obj._unflushed_counter = 0
        obj._lock = threading.Lock()
        return obj

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self._flush_every > 0:
            with self._lock:
                self._unflushed_counter += 1
                if self._unflushed_counter > self._flush_every:
                    self.flush()
                    self._unflushed_counter = 0
