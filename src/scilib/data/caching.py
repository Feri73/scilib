import inspect
import os.path
from typing import Callable, Iterable, Union, Optional, Tuple

from ..data.memmap import DiskObj
from ..utils import load_pickle, dump_pickle, Config


class Depends:
    dependants = {}

    def __init__(self, *dependants):
        self.instance_dependants = dependants

    def __call__(self, func: Callable):
        Depends.dependants[func] = self.instance_dependants
        return func


class Memoize:
    class ArrayHash:
        from ..arrays.array import Array
        import numpy as np
        import numpy.typing as npt

        def __init__(self, value: Union[npt.NDArray, Array]):
            if isinstance(value, self.Array):
                self.__value = value.numpy
            elif isinstance(value, self.np.ndarray):
                self.__value = value
            else:
                raise ValueError()

        def __eq__(self, other) -> bool:
            return (type(self) == type(other) and self.__value.shape == other.__value.shape and
                    self.np.all(self.__value == other.__value))

        def __ne__(self, other) -> bool:
            return not (self == other)

    @staticmethod
    def array_hash_func(value) -> Optional['Memoize.ArrayHash']:
        try:
            return Memoize.ArrayHash(value)
        except ValueError:
            return None

    def __init__(self, func: Callable):
        self.__func = func
        self.__path = None
        self.__name = None
        self.__pickle = None
        self.__db = None
        self.__read_cache = None
        self.__write_cache = None
        self.__verbose = None
        self.__hash_func = None

    @staticmethod
    def func_summary(func: Callable) -> str:
        try:
            if isinstance(func, Memoize):
                src = inspect.getsource(func.__func)
            else:
                src = inspect.getsource(func)
        except OSError:
            src = '1'
        if func in Depends.dependants:
            for d in Depends.dependants[func]:
                src += Memoize.func_summary(d)
        return src

    def print(self, *args, **kwargs):
        if self.__verbose:
            print(*args, **kwargs)

    @staticmethod
    def set_config(*args, **kwargs) -> Callable[[Callable], Callable]:
        def create_memoize(func: Callable):
            return Memoize(func).config(*args, **kwargs)

        return create_memoize

    def config(self, path: str, name: str, check_func_change=True, read_cache: bool = True, write_cache: bool = True,
               pickle=None, verbose=True, hash_func: Callable = None) -> Callable:
        self.__path = path
        self.__name = name
        if pickle is None:
            import pickle
        self.__pickle = pickle
        self.__read_cache = read_cache
        self.__write_cache = write_cache
        self.__verbose = verbose
        self.__hash_func = (lambda x: None) if hash_func is None else hash_func

        if self.__read_cache or self.__write_cache:
            if check_func_change:
                func_summary = self.func_summary(self.__func)
            else:
                func_summary = ''

            self.__db = DiskObj(f'{path}/{name}/db.pckl', pickle, [func_summary])
            if not check_func_change or self.__db.obj[0] != func_summary:
                hs = hash(self.__db.obj[0])
                self.print(f'Function memoized at {path}/{name} is changed. Archiving current cache with hash {hs}.')
                os.rename(f'{path}/{name}', f'{path}/{name}_{hs}')
                self.__db = DiskObj(f'{path}/{name}/db.pckl', pickle, [func_summary])
        else:
            self.print(f'Memoize disabled for {path}/{name}.')

        return self

    def __hashed_args(self, args: tuple, kwargs: dict) -> Tuple[tuple, dict]:
        new_args = []
        new_kwargs = {}
        for arg in args:
            hs = self.__hash_func(arg)
            if hs is None:
                new_args.append(arg)
            else:
                new_args.append(hs)
        for k, v in kwargs.items():
            hs = self.__hash_func(v)
            if hs is None:
                new_kwargs[k] = v
            else:
                new_kwargs[k] = hs
        return tuple(new_args), new_kwargs

    def __call__(self, *args, **kwargs):
        assert self.__path is not None and self.__name is not None and self.__pickle is not None
        hashed_args = self.__hashed_args(args, kwargs)
        if self.__read_cache:
            for i, arg_list in enumerate(self.__db.obj[1:]):
                if arg_list == hashed_args:
                    self.print(f'Match found for {args}, {kwargs} in #{i}.')
                    return load_pickle(self.__pickle, f'{self.__path}/{self.__name}/{i}.pckl')
            self.print(f'No match found for {args}, {kwargs}. Calculating the result')
        res = self.__func(*args, **kwargs)
        if self.__write_cache:
            print(f'Dumping the result in #{len(self.__db.obj) - 1}')
            dump_pickle(self.__pickle, f'{self.__path}/{self.__name}/{len(self.__db.obj) - 1}.pckl', res)
            self.__db.obj.append(hashed_args)
            self.__db.flush()
        return res


class ProgramCache:
    class Vars(Config):
        pass

    def __init__(self, path: str, verbose: bool = True, pickle=None, *flush_with):
        self.__started = False
        self.__cur_state = tuple()
        self.__path = path
        if pickle is None:
            import pickle
        self.__verbose = verbose
        self.__storage = DiskObj(path, pickle, {})
        self.__vars = self.Vars()
        self.__flush_with = list(flush_with)

    @property
    def vars(self):
        return self.__vars

    def __enter__(self) -> 'ProgramCache':
        self.__started = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        self.__started = False

    def print(self, *args, verbose: bool = None, **kwargs):
        if verbose is None:
            verbose = self.__verbose
        if verbose:
            return print(f'Program caching to {self.__path}: ', *args, **kwargs)
        
    def add_flush_with(self, flushable) -> None:
        self.__flush_with.append(flushable)

    def flush(self) -> None:
        self.__storage.flush()
        for x in self.__flush_with:
            x.flush()

    def loop(self, it: Iterable, verify_element: bool = True, cache_every: int = 1, all_steps: bool = True,
             verbose: bool = None):
        assert self.__started
        for index, element in enumerate(it):
            self.__cur_state += (index,)
            if self.__cur_state in self.__storage.obj:
                if all_steps:
                    self.__vars = self.__storage.obj[self.__cur_state][1]
                else:
                    self.__vars = self.__storage.obj['vars']
                self.print(f'Skipping index state {self.__cur_state}. Setting vars={self.__vars}.', verbose=verbose)
                if verify_element:
                    assert element == self.__storage.obj[self.__cur_state][0]
            else:
                yield element
                if all_steps:
                    self.__storage.obj[self.__cur_state] = [element if verify_element else None, self.__vars]
                else:
                    self.__storage.obj[self.__cur_state] = [element if verify_element else None, None]
                    self.__storage.obj['vars'] = self.__vars

                if index % cache_every == 0:
                    self.print(f'Flushing after processing index state {self.__cur_state}.'
                               f' Caching vars={self.__vars}.', verbose=verbose)
                    self.flush()
            self.__cur_state = self.__cur_state[:-1]

        self.flush()
