from typing import Iterable, Dict, Any, Callable, List
import numpy.typing as npt
import matplotlib
from collections import defaultdict


class PlotMemory:
    def __init__(self, func: Callable, arg_cache_inds: List[int] = None, kwarg_cache_keys: List[str] = None):
        self.__func = func
        self.__arg_cache_inds = arg_cache_inds
        self.__kwarg_cache_keys = kwarg_cache_keys
        assert arg_cache_inds is not None or kwarg_cache_keys is not None
        self.__arg_cache = defaultdict(list)
        self.__kwarg_cache = defaultdict(list)

    def __call__(self, *args, **kwargs):
        if self.__arg_cache_inds is not None:
            for i in self.__arg_cache_inds:
                self.__arg_cache[i].append(args[i])
        if self.__kwarg_cache_keys is not None:
            for k in self.__kwarg_cache_keys:
                self.__kwarg_cache[k].append(kwargs[k])

        return self.__func(*args, **kwargs)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.__arg_cache[item]
        elif isinstance(item, str):
            return self.__kwarg_cache[item]
        else:
            raise TypeError(f'item must be int or str, not {type(item)}')


def get_colors(elements: Iterable, cname='gist_rainbow') -> Dict[Any, npt.NDArray]:
    len = sum(1 for _ in elements)
    cmap = matplotlib.cm.get_cmap(cname)
    colors = {}
    for i, e in enumerate(elements):
        colors[e] = cmap(i / len)
    return colors
