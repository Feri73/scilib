import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Union, Any
import os


class Config(dict):
    def __getattr__(self, item):
        if item in self:
            return self[item]
        return super(Config, self).__getattr__(item)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def rec_update(self, other: 'Config') -> 'Config':
        for k in other:
            if k in self:
                if isinstance(self[k], Config):
                    self[k] = self[k].rec_update(other[k])
                else:
                    self[k] = other[k]
            else:
                self[k] = other[k]

        return self


class KeyDefaultDict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def rel2abs(file: str, ref__file__) -> str:
    return os.path.join(os.path.dirname(os.path.realpath(ref__file__)), file)


def do_import(file: str, element: str = None, name='_tmp_', ref__file__=None) -> Union[ModuleType, Any]:
    if ref__file__ is not None:
        file = rel2abs(file, ref__file__)
    spec = importlib.util.spec_from_file_location(name, file)
    file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(file)
    if element is None:
        return file
    else:
        return getattr(file, element)


class ProjectImports:
    def __init__(self, root_path: str, ref__file__=None):
        self.__path = root_path if ref__file__ is None else os.path.join(os.path.dirname(ref__file__), root_path)

    def __enter__(self) -> 'ProjectImports':
        sys.path.insert(0, self.__path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        sys.path = sys.path[1:]

    def in_project(self) -> bool:
        return True


def load_pickle(pickle, file_name: str):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def dump_pickle(pickle, file_name: str, obj, make_parent: bool = True) -> None:
    if make_parent:
        Path(file_name).parent.absolute().mkdir(parents=True, exist_ok=True)
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)
