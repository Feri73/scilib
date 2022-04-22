from functools import partial
from numbers import Number
from typing import Union, List, Callable, Tuple
from .utils import np, npt, NPValue, NPIndex

Axes = List[int]

ACCESSOR_STR = 'accessor'


def accessor(func: Callable) -> Callable:
    setattr(func, ACCESSOR_STR, True)
    return func


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

    def __init__(self, axes: Axes):
        self.__numpy = None
        self.__axes = axes

    @staticmethod
    def _to_ndarray(array: NPValue) -> npt.NDArray:
        return array if isinstance(array, np.ndarray) else np.array(array)

    def copy(self) -> 'ArrayView':
        return ArrayView(self.axes)

    @property
    def numpy(self) -> npt.NDArray:
        return self.__numpy

    @property
    def axes(self) -> Axes:
        return self.__axes

    def __call__(self, numpy: Union[NPValue, 'ArrayView'], axes: Axes = None):
        view = self.copy()
        view.__numpy = numpy.numpy if isinstance(numpy, ArrayView) else self._to_ndarray(numpy)
        if axes is not None:
            view.__axes = axes
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
        return tuple(np.array(self.numpy.shape)[self.axes])


class ArrayView1D(ArrayView):
    def __init__(self, axis: int):
        super().__init__([axis])

    def copy(self) -> 'ArrayView1D':
        return ArrayView1D(self.axes[0])

    def __call__(self, numpy: Union[NPValue, 'ArrayView'], axes: Union[Axes, int] = None):
        if isinstance(axes, int):
            axes = [axes]
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
    def apply(self, func: Callable, **kwargs):
        return self(func(self.numpy, axis=self.axis, **kwargs))


class SampledTimeView(ArrayView1D):
    def __init__(self, axis: int, freq: float, start_time: float = 0.):
        super().__init__(axis)

        self.__freq = freq
        self.__start_time = start_time

    def _new(self, axis: int, freq: float, start_time: float, use_start_time: bool) -> 'SampledTimeView':
        return SampledTimeView(axis, freq, start_time if use_start_time else 0.)

    def copy(self) -> 'SampledTimeView':
        return self._new(self.axis, self.freq, self.start_time, True)

    @property
    def freq(self) -> float:
        return self.__freq

    @property
    def start_time(self) -> float:
        return self.__start_time

    @property
    def times(self):
        return np.arange(self.start_time, (self.start_time * self.freq + len(self)) / self.freq, 1 / self.freq)

    @accessor
    def __getitem__(self, times: NPIndex) -> Union[ArrayView1D, 'SampledTimeView']:
        if isinstance(times, Number):
            times = int(self.freq * (times - self.start_time))
            if times < 0:
                raise ValueError()
        elif isinstance(times, slice):
            orig_slice = times
            step = times.step if times.step is None else times.step * self.freq
            if step is not None and int(step) < step:
                times = np.arange(times.start or self.start_time, times.stop or len(self) / self.freq, times.step)
                return self._new(self.axis, step / self.freq, times[0], False)(self[times])
            else:
                times = slice(times.start if times.start is None else int((times.start - self.start_time) * self.freq),
                              times.stop if times.stop is None else int((times.stop - self.start_time) * self.freq),
                              step if step is None else int(step))
                if times.start is not None and times.start < 0:
                    raise ValueError()
        else:
            times = [int((t - self.start_time) * self.freq) for t in times]
            for t in times:
                if t < 0:
                    raise ValueError()
        res = super(SampledTimeView, self).__getitem__(times)
        if isinstance(times, slice):
            return self._new(self.axis, 1 / (orig_slice.step or 1 / self.freq),
                             orig_slice.start or self.start_time, False)(res)
        else:
            return ArrayView1D(self.axis)(res)

    @accessor
    def take(self, start_time: float, duration: float, freq: float = None,
             set_start_time: bool = False) -> 'SampledTimeView':
        freq = freq or self.freq
        start_ind = int(start_time * self.freq)
        inds = []
        while len(inds) < duration * freq:
            inds.append(start_ind)
            start_ind += int(self.freq / freq)
        return self._new(self.axis, freq, start_time, set_start_time)(super(SampledTimeView, self).__getitem__(inds))

    @accessor
    def start_at(self, start_time: float):
        return self._new(self.axis, self.freq, start_time, True)(self.numpy)

    def __setitem__(self, times: NPIndex, value: NPValue) -> None:
        if isinstance(times, Number):
            times = int(self.freq * times)
        elif isinstance(times, slice):
            times = slice(times.start if times.start is None else int(times.start * self.freq),
                          times.stop if times.stop is None else int(times.stop * self.freq),
                          times.step if times.step is None else int(times.step * self.freq))
        else:
            times = [int(t * self.freq) for t in times]
        super(SampledTimeView, self).__setitem__(times, value)

    def conv_weight(self, duration: float, per_sample: bool = True) -> NPValue:
        return 1 / (duration * (self.freq if per_sample else 1)) * np.ones(int(duration * self.freq))


class EventTimeView(ArrayView1D):
    def __init__(self, axis: int, times: NPIndex):
        super().__init__(axis)
        self.__times = self._to_ndarray(times)

    @property
    def times(self):
        return self.__times

    def copy(self) -> 'EventTimeView':
        return EventTimeView(self.axis, self.times)

    def __call__(self, numpy: Union[NPValue, 'ArrayView'], axes: Union[Axes, int] = None):
        if isinstance(axes, int):
            axes = [axes]
        axes = axes or [self.axis]
        if isinstance(numpy, ArrayView):
            assert len(axes) == 1 and len(numpy) == len(self.times)
        else:
            numpy = self._to_ndarray(numpy)
            assert len(axes) == 1 and numpy.shape[axes[0]] == len(self.times)
        return super(EventTimeView, self).__call__(numpy, axes)

    @accessor
    def to_sampled(self, freq: float, default_value: NPValue = None,
                   start_time: float = 0, end_time: float = None, set_start_time: bool = False) -> SampledTimeView:
        assert min(self.times[1:] - self.times[:-1]) >= 1 / freq

        default_value = default_value or [0]

        slcs = [slice(None)] * len(self.numpy.shape)

        res = []
        t = start_time
        next_event_i = 0
        end_time = end_time or self.times[-1]
        while t <= end_time:
            if next_event_i <= len(self.times) - 1 and self.times[next_event_i] <= t:
                slcs[self.axis] = [next_event_i]
                res.append(self.numpy[tuple(slcs)])
                next_event_i += 1
            else:
                res.append(default_value if len(res) == 0 else np.ones_like(res[0]) * default_value)
            t += 1 / freq
        return SampledTimeView(self.axis, freq, start_time if set_start_time else 0.)(
            np.swapaxes(res, 0, self.axis + 1)[0])

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
        return EventTimeView(self.axis, times)(data)


class KeyView(ArrayView1D):
    def __init__(self, axis: int, *keys: str):
        super().__init__(axis)

        assert sorted(keys) == sorted(set(keys))

        self.__keys = keys
        self.__reverse_keys = {k: i for i, k in enumerate(keys)}

    def copy(self) -> 'KeyView':
        return KeyView(self.axis, *self.__keys)

    @property
    def keys(self):
        return self.__keys

    @accessor
    def __getitem__(self, keys: Union[str, List[str]]) -> 'KeyView':
        if isinstance(keys, str):
            keys = [keys]
        keys = sum([[x for x in self.__keys if x.startswith(key)] for key in keys], [])

        return KeyView(self.axis, *keys)(super(KeyView, self).__getitem__([self.__reverse_keys[key] for key in keys]))

    def __setitem__(self, keys: Union[str, List[str]], value: NPValue) -> None:
        if isinstance(keys, str):
            keys = [keys]
        keys = sum([[x for x in self.__keys if x.startswith(key)] for key in keys], [])
        super(KeyView, self).__setitem__([self.__reverse_keys[key] for key in keys], value)


class Array:
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

    def __init__(self, **views: ArrayView):
        axes = []
        self.views = views
        for view_name, view in views.items():
            for axis in axes:
                assert axis not in view.axes
            axes += view.axes
            setattr(self, view_name, Array.View(view_name, self))

    def __call__(self, numpy: NPValue) -> 'Array':
        numpy = ArrayView._to_ndarray(numpy)
        return Array(**{view_name: view(numpy) for view_name, view in self.views.items()})

    @property
    def numpy(self) -> npt.NDArray:
        numpy = None
        for n in map(lambda x: x.numpy, self.views.values()):
            assert numpy is None or n is numpy
            numpy = n
        return numpy

    def reduce(self, func: Callable, *views: str, keepdims=False, **kwargs) -> 'Array':
        axes = []
        for view_name in views:
            axes += self.views[view_name].axes
        res = self(func(self.numpy, axis=tuple(axes), keepdims=True, **kwargs))
        if not keepdims:
            res = res.squeeze(*views)
        return res

    def squeeze(self, *views: str) -> 'Array':
        axes = []
        for view_name in views:
            if np.any(np.array(self.views[view_name].shape) != 1):
                raise ValueError()
            axes += self.views[view_name].axes
        numpy = np.squeeze(self.numpy, tuple(axes))
        views = {n: v(numpy, [a - len([aa for aa in axes if aa < a]) for a in v.axes])
                 for n, v in self.views.items() if n not in views}
        return Array(**views)

    def expand(self, **views: ArrayView) -> 'Array':
        new_axes = []
        for view_name, view in views.items():
            new_axes += view.axes
        numpy = np.expand_dims(self.numpy, new_axes)
        for view_name, view in views.items():
            views[view_name] = view(numpy)
        for view_name, view in self.views.items():
            views[view_name] = view(numpy, [a + len([aa for aa in new_axes if aa <= a]) for a in view.axes])
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
                view = ArrayView1D(self.ndim)
                view_name = '__tmp__'
            elif len(views) == 0 and view_name is not None:
                view = ArrayView1D(self.ndim)
            else:
                view_name, view = list(views.items())[0]
            arrays = tuple(array.expand(**{view_name: view}) for array in arrays)
            if view_name == '__tmp__':
                for array in arrays:
                    delattr(array, view_name)
            self = arrays[0]

        return self(np.concatenate(tuple(x.numpy for x in arrays), axis=view.axis))
