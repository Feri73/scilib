import inspect
import os.path
from pathlib import Path
from typing import Callable

from scilib.utils import load_pickle, dump_pickle


class Depends:
    dependants = {}

    def __init__(self, *dependants):
        self.instance_dependants = dependants

    def __call__(self, func: Callable):
        Depends.dependants[func] = self.instance_dependants
        return func


class Memoize:
    def __init__(self, func: Callable):
        self.__func = func
        self.__path = None
        self.__name = None
        self.__pickle = None
        self.__db = None
        self.__read_cache = None
        self.__write_cache = None
        self.__verbose = None

    @staticmethod
    def func_summary(func: Callable) -> str:
        try:
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

    def config(self, path: str, name: str, check_func_change=True, read_cache: bool = True, write_cache: bool = True,
               pickle=None, verbose=True) -> Callable:
        self.__path = path
        self.__name = name
        if pickle is None:
            import pickle
        self.__pickle = pickle
        self.__read_cache = read_cache
        self.__write_cache = write_cache
        self.__verbose = verbose

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

        return self.__call__

    def __call__(self, *args, **kwargs):
        assert self.__path is not None and self.__name is not None and self.__pickle is not None
        if self.__read_cache:
            for i, arg_list in enumerate(self.__db.obj[1:]):
                if arg_list == [args, kwargs]:
                    self.print(f'Match found for {args}, {kwargs} in #{i}.')
                    return load_pickle(self.__pickle, f'{self.__path}/{self.__name}/{i}.pckl')
            self.print(f'No match found for {args}, {kwargs}. Calculating the result')
        res = self.__func(*args, **kwargs)
        if self.__write_cache:
            print(f'Dumping the result in #{len(self.__db.obj) - 1}')
            dump_pickle(self.__pickle, f'{self.__path}/{self.__name}/{len(self.__db.obj) - 1}.pckl', res)
            self.__db.obj.append([args, kwargs])
            self.__db.store()
        return res


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

    def store(self) -> None:
        Path(self.__path).parent.absolute().mkdir(parents=True, exist_ok=True)
        dump_pickle(self.__pickle, self.__path, self.__obj)
