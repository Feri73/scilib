from functools import partial
from numbers import Number
from types import ModuleType
from typing import Union, List, Callable, Tuple, Dict
from math import floor

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from .utils import numpy_lib, npt, NPValue, NPIndex
from .. import arrays

Axes = List[int]

ACCESSOR_STR = 'accessor'


def accessor(func: Callable) -> Callable:
    setattr(func, ACCESSOR_STR, True)
    return func


def _compute(x: Union[NPValue, Tuple[NPValue]], *, np: ModuleType) -> Union[NPValue, Tuple[NPValue]]:
    if hasattr(np, 'compute'):
        res = np.compute(x)
        if isinstance(res, tuple) and len(res) == 1:
            return res[0]
        return res
    return x


class ArrayViewMeta(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        accessors = []
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if hasattr(attr, ACCESSOR_STR):
                accessors.append(attr_name)
        cls.accessors = accessors
        return cls


class ArrayView(metaclass=ArrayViewMeta):
    accessors = []

    def __init__(self, axes: Axes, *, np: ModuleType):
        self.__numpy = None
        self.__axes = axes
        self.__np = np

    @staticmethod
    def _to_ndarray(array: NPValue, *, np: ModuleType) -> npt.NDArray:
        if np == numpy_lib and isinstance(array, np.memmap):
            return array
        return np.asarray(array)  # if isinstance(array, np.ndarray) else np.array(array)

    def copy(self, axes: Axes = None, *, np: ModuleType = None) -> 'ArrayView':
        return ArrayView(self.axes if axes is None else axes.copy(), np=self.np if np is None else np)

    @property
    def np(self) -> ModuleType:
        return self.__np

    @property
    def numpy(self) -> npt.NDArray:
        return self.__numpy

    @property
    def axes(self) -> Axes:
        return self.__axes

    def __call__(self, numpy: Union[NPValue, 'ArrayView'], axes: Axes = None):
        view = self.copy(axes)
        view.__numpy = numpy.numpy if isinstance(numpy, ArrayView) else self._to_ndarray(numpy, np=self.np)
        return view

    @accessor
    def __getitem__(self, indices: NPIndex) -> 'ArrayView':
        slcs = [slice(None)] * len(self.__numpy.shape)
        if isinstance(indices, tuple):
            for axis, index in zip(self.axes, indices):
                slcs[axis] = index
        else:
            for axis in self.axes:
                slcs[axis] = indices
        for slc_i, slc in enumerate(slcs):
            if isinstance(slc, Number):
                slcs[slc_i] = [slc]
        return self(self.numpy[tuple(slcs)])

    def __setitem__(self, indices: NPIndex, value: NPValue) -> None:
        slcs = [slice(None)] * len(self.__numpy.shape)
        if isinstance(indices, tuple):
            for axis, index in zip(self.axes, indices):
                slcs[axis] = index
        else:
            for axis in self.axes:
                slcs[axis] = indices
        self.numpy[tuple(slcs)] = value

    @accessor
    def reduce(self, func: Callable, **kwargs):
        return self(func(self.numpy, axis=tuple(self.axes), keepdims=True, **kwargs))

    @accessor
    def apply(self, func: Callable, **kwargs):
        return self(func(self.numpy, axis=tuple(self.axes), **kwargs))

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(numpy_lib.array(self.numpy.shape)[self.axes])


class ArrayView1D(ArrayView):
    def __init__(self, axis: int, *, np: ModuleType = numpy_lib):
        super().__init__([axis], np=np)

    def copy(self, axes: Union[Axes, int] = None, *, np: ModuleType = None) -> 'ArrayView1D':
        axes = self._correct_axes(axes)
        return ArrayView1D((self.axes if axes is None else axes.copy())[0], np=self.np if np is None else np)

    def __call__(self, numpy: Union[NPValue, 'ArrayView'], axes: Union[Axes, int] = None):
        axes = self._correct_axes(axes)
        return super(ArrayView1D, self).__call__(numpy, axes)

    @property
    def axis(self) -> int:
        return self.axes[0]

    def __len__(self):
        return self.numpy.shape[self.axis]

    @accessor
    def reduce(self, func: Callable, **kwargs):
        return self(func(self.numpy, axis=self.axis, keepdims=True, **kwargs))

    @accessor
    def appended(self, value: NPValue) -> 'ArrayView1D':
        value = self._to_ndarray(value, np=self.np)
        if (self.numpy.shape[:self.axis] + self.numpy.shape[self.axis + 1:]) == value.shape:
            value = self.np.expand_dims(value, self.axis)
        else:
            assert self.numpy.shape[:self.axis] == value.shape[:self.axis]
            assert self.numpy.shape[self.axis + 1:] == value.shape[self.axis + 1:]
        return self(self.np.append(self.numpy, value, axis=self.axis))

    @accessor
    def apply(self, func: Callable, **kwargs):
        return self(func(self.numpy, axis=self.axis, **kwargs))

    @staticmethod
    def _correct_axes(axes: Union[int, Axes]) -> Axes:
        if isinstance(axes, int):
            axes = [axes]
        return axes


class SampledTimeView(ArrayView1D):
    def __init__(self, axis: int, freq: float, start_time: float = 0., *, np: ModuleType = numpy_lib):
        super().__init__(axis, np=np)

        self.__freq = freq
        self.__start_time = start_time
        self.__times = None

    def _new(self, axis: int, freq: float, start_time: float, use_start_time: bool,
             *, np: ModuleType = None) -> 'SampledTimeView':
        return SampledTimeView(axis, freq, start_time if use_start_time else 0., np=self.np if np is None else np)

    def copy(self, axes: Union[int, Axes] = None, *, np: ModuleType = None) -> 'SampledTimeView':
        axes = self._correct_axes(axes)
        return self._new((self.axes if axes is None else axes.copy())[0], self.freq, self.start_time, True, np=np)

    @property
    def freq(self) -> float:
        return self.__freq

    @property
    def start_time(self) -> float:
        return self.__start_time

    @property
    def times(self):
        if self.__times is None:
            self.__times = (self.start_time, self.freq, len(self),
                            self.np.arange(self.start_time,
                                           (self.start_time * self.freq + len(self)) / self.freq,
                                           1 / self.freq)[:len(self)])
        if self.__times[:-1] == (self.start_time, self.freq, len(self)):
            return self.__times[-1]
        self.__times = None
        return self.times

    def __linear_interpolation(self, indices: NPIndex, floor_res: NPValue = None, ceil_res: NPValue = None) -> NPValue:
        indices = _compute(indices, np=self.np)
        floor_inds = self._to_ndarray(self.np.floor(indices), np=self.np)
        ceil_inds = floor_inds + 1
        if floor_res is None:
            floor_res = super(SampledTimeView, self).__getitem__(floor_inds.astype(self.np.int32)).numpy
        if ceil_res is None:
            ceil_res = super(SampledTimeView, self).__getitem__(ceil_inds.astype(self.np.int32)).numpy
        ceil_inds = self.np.reshape(ceil_inds, [1 if i != self.axis else -1 for i in range(len(self.numpy.shape))])
        floor_inds = self.np.reshape(floor_inds, [1 if i != self.axis else -1 for i in range(len(self.numpy.shape))])
        indices = self.np.reshape(self._to_ndarray(indices, np=self.np),
                                  [1 if i != self.axis else -1 for i in range(len(self.numpy.shape))])
        res = floor_res + (ceil_res - floor_res) / (ceil_inds - floor_inds) * (indices - floor_inds)
        res = self.np.where(self.np.isnan(res), floor_res, res)
        return res.astype(self.numpy.dtype)

    @accessor
    @arrays.VERSION.check_assumption(sampeld_time_view_interpolation='linear')
    def __getitem__(self, times: NPIndex) -> Union[ArrayView1D, 'SampledTimeView']:
        if isinstance(times, Number):
            indices = self.freq * (times - self.start_time)
            if indices < 0:
                raise ValueError()
            indices = self.np.array([indices])
        elif isinstance(times, slice):
            if times.step is not None and times.step < 0:
                raise ValueError()
            freq = self.freq if times.step is None else 1. / times.step
            step = 1. / self.freq if times.step is None else times.step
            start_time = self.start_time if times.start is None else times.start
            end_time = float(_compute(self.times[-1], np=self.np)) if times.stop is None else times.stop
            if start_time < self.start_time:
                raise ValueError()
            if step % (1. / self.freq) == 0:
                if (start_time - self.start_time) % (1. / self.freq) == 0:
                    res = super(SampledTimeView, self).__getitem__(
                        slice(round((start_time - self.start_time) * self.freq),
                              round((end_time - self.start_time) * self.freq) + 1, round(step * self.freq)))
                else:
                    start_idx = (start_time - self.start_time) * self.freq
                    end_idx = (end_time - self.start_time) * self.freq
                    step_idx = round(step * self.freq)
                    floor_res = super(SampledTimeView, self).__getitem__(
                        slice(floor(start_idx), floor(end_idx) + 1, step_idx))
                    ceil_res = super(SampledTimeView, self).__getitem__(
                        slice(floor(start_idx) + 1, floor(end_idx) + 1 + 1, step_idx))
                    min_len = min(len(floor_res), len(ceil_res))
                    floor_res = super(SampledTimeView, floor_res).__getitem__(slice(min_len))
                    floor_res = floor_res.numpy
                    ceil_res = ceil_res.numpy
                    frac = start_idx - floor(start_idx)
                    res = (floor_res * (1 - frac) + ceil_res * frac).astype(self.numpy.dtype)
            else:
                times = self.np.arange(start_time, end_time, step)
                res = self[times]
            return self._new(self.axis, freq, start_time, False)(res)
        else:
            indices = (self.np.array(times) - self.start_time) * self.freq
            if bool(_compute(self.np.any(indices < 0), np=self.np)):
                raise ValueError()
        ceil_inds = self.np.floor(indices) + 1
        valid_ceil_inds_is = self.np.asarray(ceil_inds < len(self))
        indices = indices[valid_ceil_inds_is]
        res = self.__linear_interpolation(indices)
        return ArrayView1D(self.axis, np=self.np)(res)

    @arrays.VERSION.check_assumption(sampeld_time_view_inds='non-neg;+s_time')
    def __setitem__(self, times: NPIndex, value: NPValue) -> None:
        if isinstance(times, Number):
            times = int(self.freq * (times - self.start_time))
            if times < 0:
                raise ValueError()
        elif isinstance(times, slice):
            if times.step is not None and times.step < 0:
                raise ValueError()
            step = times.step if times.step is None else times.step * self.freq
            if step is not None and int(step) < step:
                raise ValueError()
            else:
                times = slice(times.start if times.start is None else int((times.start - self.start_time) * self.freq),
                              times.stop if times.stop is None else int((times.stop - self.start_time) * self.freq),
                              step if step is None else int(step))
                if times.start is not None and times.start < 0 or times.stop < 0:
                    raise ValueError()
        else:
            times = [int((t - self.start_time) * self.freq) for t in times]
            for t in times:
                if t < 0:
                    raise ValueError()
        super(SampledTimeView, self).__setitem__(times, value)

    @accessor
    def take(self, start_time: float = None, duration: float = None, freq: float = None,
             set_start_time: bool = False) -> 'SampledTimeView':
        start_time = self.start_time if start_time is None else start_time
        duration = self.times[-1] - self.times[0] if duration is None else duration
        freq = self.freq if freq is None else freq
        res = self[start_time:start_time + duration:1. / freq]
        return self._new(self.axis, freq, start_time, set_start_time)(res)

    @accessor
    def start_at(self, start_time: float):
        return self._new(self.axis, self.freq, start_time, True)(self.numpy)

    def conv_weight(self, duration: float, per_sample: bool = True) -> NPValue:
        return 1 / (duration * (self.freq if per_sample else 1)) * self.np.ones(int(duration * self.freq))


class EventTimeView(ArrayView1D):
    def __init__(self, axis: int, times: NPIndex, *, np: ModuleType = numpy_lib):
        super().__init__(axis, np=np)
        self.__times = self._to_ndarray(times, np=self.np)

    @property
    def times(self):
        return self.__times

    def copy(self, axes: Union[int, Axes] = None, *, np: ModuleType = None) -> 'EventTimeView':
        axes = self._correct_axes(axes)
        return EventTimeView((self.axes if axes is None else axes.copy())[0], self.times,
                             np=self.np if np is None else np)

    def __call__(self, numpy: Union[NPValue, 'ArrayView'], axes: Union[Axes, int] = None):
        axes = self._correct_axes(axes)
        axes = [self.axis] if axes is None else axes
        if isinstance(numpy, ArrayView):
            assert len(axes) == 1 and numpy.shape[axes[0]] == len(self.times)
        else:
            numpy = self._to_ndarray(numpy, np=self.np)
            assert len(axes) == 1 and numpy.shape[axes[0]] == len(self.times)
        return super(EventTimeView, self).__call__(numpy, axes)

    @accessor
    def to_sampled(self, freq: float, default_value: Union[NPValue, Literal['last']],
                   start_time: float = 0., end_time: float = None, set_start_time: bool = False,
                   multi_events: Literal['raise', 'mean', 'sum'] = 'raise') -> SampledTimeView:
        """
        :param default_value: if 'last', use the last value of the array or zero if it is the start of the array
        :param multi_events: what to do if multiple events are in the same period
                'raise': check and raise exception
                'mean': get mean
                'sum': get sum
        """
        assert multi_events in ('raise', 'mean', 'sum')

        if multi_events != 'sum' or default_value != 0.:
            raise NotImplementedError()

        times = self.times
        data = self.np.moveaxis(self.numpy, self.axis, -1)  # bring event axis to last

        end_time = times[-1] if end_time is None else end_time
        n_steps = int(_compute(self.np.floor((end_time - start_time) * freq) + 1, np=self.np))
        sample_times = start_time + self.np.arange(n_steps) / freq

        bin_idx = self.np.searchsorted(sample_times, times, side='left')
        flat_data = self.np.reshape(data, (-1, data.shape[-1]))
        out = self.np.stack([self.np.bincount(bin_idx, flat_data[i], minlength=n_steps).astype(self.numpy.dtype)
                             for i in range(len(flat_data))])
        out = self.np.reshape(out, (*data.shape[:-1], n_steps))
        out = self.np.moveaxis(out, -1, self.axis)

        return SampledTimeView(
            self.axis,
            freq,
            start_time if set_start_time else 0.,
            np=self.np
        )(out)

    @accessor
    def __getitem__(self, indices: NPIndex) -> 'EventTimeView':
        slcs = [slice(None)] * len(self.numpy.shape)
        slcs[self.axis] = indices
        data = self.numpy[tuple(slcs)]
        times = self.times[indices]
        if len(data.shape) != len(slcs):
            slcs[self.axis] = None
            data = data[tuple(slcs)]
            times = [times]
        return EventTimeView(self.axis, times, np=self.np)(data)


class KeyView(ArrayView1D):
    def __init__(self, axis: int, *keys: str, np: ModuleType = numpy_lib):
        super().__init__(axis, np=np)

        assert sorted(keys) == sorted(set(keys))

        self.__keys = keys
        self.__reverse_keys = {k: i for i, k in enumerate(keys)}

    def __call__(self, numpy: Union[NPValue, 'ArrayView'], axes: Union[Axes, int] = None):
        axes = self._correct_axes(axes)
        axes = [self.axis] if axes is None else axes
        if isinstance(numpy, ArrayView):
            assert len(axes) == 1 and numpy.numpy.shape[axes[0]] == len(self.keys)
        else:
            numpy = self._to_ndarray(numpy, np=self.np)
            assert len(axes) == 1 and numpy.shape[axes[0]] == len(self.keys)
        return super(KeyView, self).__call__(numpy, axes)

    def _list_to_slice(self, lst):
        if not lst:
            return lst
        assert all([x >= 0 for x in lst])
        step = lst[1] - lst[0] if len(lst) > 1 else 1
        for i in range(len(lst) - 1):
            if lst[i + 1] - lst[i] != step:
                return lst
        start = lst[0]
        stop = lst[-1] + step
        if stop < 0:
            stop = None
        return slice(start, stop, step)

    def copy(self, axes: Union[int, Axes] = None, *, np: ModuleType = None) -> 'KeyView':
        axes = self._correct_axes(axes)
        return KeyView((self.axes if axes is None else axes.copy())[0], *self.__keys, np=self.np if np is None else np)

    @property
    def keys(self):
        return self.__keys

    @arrays.VERSION.check_assumption(key_view_plain_pattern='equality')
    def __update_keys(self, pattern: str, keys: List[str]) -> List[str]:
        if pattern.startswith('!<'):
            return [x for x in keys if pattern[2:] not in x]
        if pattern.startswith('!$'):
            return [x for x in keys if not x.startswith(pattern[2:])]
        if pattern.startswith('!') and pattern.endswith('$'):
            return [x for x in keys if not x.endswith(pattern[1:-1])]
        if pattern.startswith('!'):
            return [x for x in keys if x != pattern[1:]]
        if pattern.startswith('<'):
            return keys + [x for x in self.__keys if pattern[1:] in x]
        if pattern.startswith('$'):
            return keys + [x for x in self.__keys if x.startswith(pattern[1:])]
        if pattern.endswith('$'):
            return keys + [x for x in self.__keys if x.endswith(pattern[:-1])]
        else:
            return keys + [x for x in self.__keys if x == pattern]

    def infer_keys(self, keys: Union[str, List[str]]) -> List[str]:
        if isinstance(keys, str):
            keys = [keys]
        if len(keys) == 0:
            return keys
        exclude_patterns = []
        has_include_pattern = False
        final_keys = []
        for pattern in keys:
            if pattern.startswith('!') or pattern.endswith('!'):
                exclude_patterns.append(pattern)
            else:
                has_include_pattern = True
                final_keys = self.__update_keys(pattern, final_keys)
        if not has_include_pattern:
            final_keys = self.__keys
        for pattern in exclude_patterns:
            final_keys = self.__update_keys(pattern, final_keys)
        return final_keys

    @accessor
    def __getitem__(self, keys: Union[str, List[str]]) -> 'KeyView':
        """
        needs to return elements in the order keys are given
        """
        keys = self.infer_keys(keys)
        new_view = self.copy()
        new_view = new_view(self)
        new_view.__keys = keys
        return super(KeyView, new_view).__getitem__(self._list_to_slice([self.__reverse_keys[key] for key in keys]))

    @accessor
    def appended(self, value: NPValue, *keys: str) -> 'KeyView':
        value = self._to_ndarray(value, np=self.np)
        assert all([key not in self.__keys for key in keys])
        if (self.numpy.shape[:self.axis] + self.numpy.shape[self.axis + 1:]) == value.shape:
            value = self.np.expand_dims(value, self.axis)
        else:
            assert self.numpy.shape[:self.axis] == value.shape[:self.axis]
            assert self.numpy.shape[self.axis + 1:] == value.shape[self.axis + 1:]
        assert len(keys) == value.shape[self.axis]

        new_view = self.copy()
        new_view = new_view(self)
        new_view.__keys = [*self.__keys, *keys]
        return super(KeyView, new_view).appended(value)

    def __setitem__(self, keys: Union[str, List[str]], value: NPValue) -> None:
        """
        needs to return elements in the order keys are given
        """
        keys = self.__infer_keys(keys)
        super(KeyView, self).__setitem__([self.__reverse_keys[key] for key in keys], value)


class ArrayMeta(type):
    operators = ['add', 'sub', 'mul', 'floordiv', 'truediv', 'mod', 'pow', 'lshift', 'rshift', 'and', 'xor', 'or',
                 'radd', 'rsub', 'rmul', 'rfloordiv', 'rtruediv', 'rmod', 'rpow', 'rlshift', 'rrshift', 'rand',
                 'rxor', 'ror',
                 'neg', 'abs', 'invert', 'complex', 'int', 'long', 'float', 'oct', 'hex',
                 'lt', 'le', 'eq', 'ne', 'ge', 'gt']
    other_methods = ['astype']

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        for method in ArrayMeta.operators:
            cls.set_arithmetic_method(f'__{method}__')
        for method in ArrayMeta.other_methods:
            cls.set_arithmetic_method(method)
        return cls

    def set_arithmetic_method(cls, method: str):
        def op(self, *args, **kwargs):
            args = [a.numpy if isinstance(a, Array) else a for a in args]
            kwargs = {k: (a.numpy if isinstance(a, Array) else a) for k, a in kwargs.items()}
            return self(getattr(self.numpy, method)(*args, **kwargs))

        setattr(cls, method, op)


class Array(metaclass=ArrayMeta):
    class View:
        def __init__(self, name: str, array: 'Array'):
            self.__name = name
            self.__array = array

            for accessor in array.views[name].accessors:
                setattr(self, accessor, partial(self._return_accessor, accessor=accessor))

        @property
        def view(self) -> ArrayView:
            return self.__array.views[self.__name]

        def _return_accessor(self, *args, accessor: str, **kwargs) -> 'Array':
            value = getattr(self.view, accessor)(*args, **kwargs)
            views = {self.__name: value}
            for view_name, view in self.__array.views.items():
                if view_name != self.__name:
                    views[view_name] = view(value)
            return Array(**views)

        def __getitem__(self, item) -> 'Array':
            return self._return_accessor(item, accessor='__getitem__')

    class ValueSetter:
        def __init__(self, array: 'Array', **indices: NPIndex):
            self.__array = array
            self.__indices = indices

        def __setattr__(self, name, value):
            if name == 'value':
                inds_array = self.__array(self.__array.np.arange(self.__array.numpy.size).reshape(self.__array.shape))
                for view_name, ind in self.__indices.items():
                    inds_array = getattr(inds_array, view_name)[ind]
                self.__array.numpy[self.__array.np.unravel_index(inds_array.numpy.flat[:],
                                                                 self.__array.numpy.shape)] = \
                    ArrayView._to_ndarray(value, np=self.__array.np).flat[:]
            super(Array.ValueSetter, self).__setattr__(name, value)

    def __init__(self, **views: ArrayView):
        self.__np = None
        axes = []
        self.__views = views
        for view_name, view in views.items():
            if self.__np is None:
                self.__np = view.np
            else:
                assert self.__np is view.np
            for axis in axes:
                assert axis not in view.axes
            axes += view.axes
            setattr(self, view_name, Array.View(view_name, self))

    @property
    def views(self) -> Dict[str, ArrayView]:
        return self.__views.copy()

    def __call__(self, numpy: NPValue) -> 'Array':
        numpy = ArrayView._to_ndarray(numpy, np=self.np)
        return Array(**{view_name: view(numpy) for view_name, view in self.views.items()})

    def copy(self, *, np: ModuleType = None, **views: ArrayView) -> 'Array':
        kept_views = views.copy()
        for view_name, view in self.views.items():
            if view_name in kept_views:
                assert view.axes == kept_views[view_name].axes
            elif view.axes not in [v.axes for v in kept_views.values()]:
                kept_views[view_name] = view
        return Array(**{view_name: view.copy(np=np) for view_name, view in kept_views.items()})

    def __bool__(self):
        return self.numpy.__bool__()

    def setter(self, **indices: NPIndex) -> ValueSetter:
        return self.ValueSetter(self, **indices)

    @property
    def np(self) -> ModuleType:
        return numpy_lib if self.__np is None else self.__np

    @property
    def numpy(self) -> npt.NDArray:
        numpy = None
        for n in map(lambda x: x.numpy, self.views.values()):
            assert numpy is None or n is numpy
            numpy = n
        return numpy

    def o_numpy(self, *views: str) -> npt.NDArray:
        return self.np.transpose(self.numpy, sum((self.views[view_name].axes for view_name in views), []))

    def reduce(self, func: Callable, *views: str, keepdims=False, **kwargs) -> 'Array':
        axes = []
        for view_name in views:
            axes += self.views[view_name].axes
        res = Array(**{view_name: ArrayView(view.axes, np=self.np) for view_name, view in self.views.items()})(
            func(self.numpy, axis=tuple(axes), keepdims=True, **kwargs))
        if not keepdims:
            res = res.squeeze(*views)
        numpy = res.numpy
        for view_name, view in res.views.items():
            if view_name in self.views:
                if view_name in views:
                    res.__views[view_name] = ArrayView(view.axes, np=self.np)
                else:
                    res.__views[view_name] = self.views[view_name].copy(view.axes)
        return res(numpy)

    def squeeze(self, *views: str, all_ones: bool = False) -> 'Array':
        if len(views) == 1 and views[0] is True:
            views = []
            all_ones = True
        assert not all_ones or len(views) == 0

        if all_ones:
            views = [view_name for view_name, view in self.views.items()
                     if numpy_lib.array(self.views[view_name].shape) == 1]

        axes = []
        for view_name in views:
            if numpy_lib.any(numpy_lib.array(self.views[view_name].shape) != 1):
                raise ValueError()
            axes += self.views[view_name].axes
        numpy = self.np.squeeze(self.numpy, tuple(axes))
        views = {n: v(numpy, [a - len([aa for aa in axes if aa < a]) for a in v.axes])
                 for n, v in self.views.items() if n not in views}
        return Array(**views)

    def expand(self, **views: ArrayView) -> 'Array':
        new_axes = []
        for view_name, view in views.items():
            new_axes += view.axes
        numpy = self.np.expand_dims(self.numpy, new_axes)
        for view_name, view in views.items():
            views[view_name] = view(numpy)
        for view_name, view in self.views.items():
            views[view_name] = view(numpy, [a + len([aa for aa in new_axes if aa <= a]) for a in view.axes])
        return Array(**views)

    def transpose(self, *views_name: str) -> 'Array':
        assert set(views_name) == set(self.views.keys()) and len(views_name) == len(self.views)
        axes = []
        new_view_axes = {}
        for view_name in views_name:
            new_view_axes[view_name] = [len(axes) + i for i in range(len(self.views[view_name].axes))]
            axes += self.views[view_name].axes
        numpy = self.np.transpose(self.numpy, axes)
        views = {n: v(numpy, new_view_axes[n]) for n, v in self.views.items()}
        return Array(**views)


    @property
    def shape(self) -> Tuple[int, ...]:
        return self.numpy.shape

    @property
    def ndim(self) -> int:
        return self.numpy.ndim

    @staticmethod
    def concatenate(arrays: Tuple['Array', ...], view_name: str = None, **views: ArrayView1D) -> 'Array':
        self = arrays[0]
        if view_name in self.views:
            if len(views) != 0:
                raise ValueError()
            view = self.views[view_name]
            if not isinstance(view, ArrayView1D):
                raise TypeError()
        else:
            if len(views) == 0 and view_name is None:
                view = ArrayView1D(self.ndim, np=self.np)
                view_name = '__tmp__'
            elif len(views) == 0 and view_name is not None:
                view = ArrayView1D(self.ndim, np=self.np)
            else:
                view_name, view = list(views.items())[0]
            arrays = tuple(array.expand(**{view_name: ArrayView1D(view.axis, np=self.np)}) for array in arrays)
            if view_name == '__tmp__':
                for array in arrays:
                    delattr(array, view_name)
            self = arrays[0]
            self.__views[view_name] = view

        return self(self.np.concatenate(tuple(x.numpy for x in arrays), axis=view.axis))
