import importlib.util
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Union, Any, Callable
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


class BadAssumption(Exception):
    pass


class Version:
    SUPPRESS = False

    def __init__(self, unit_name: str, **unit_assumptions):
        """
        used by each unit to specify assumptions about the version of the unit
        """
        self.unit_assumptions = unit_assumptions
        self.user_assumptions = {}
        self.unit_name = unit_name
        self.suppressed = False

    def suppress(self) -> None:
        self.suppressed = True

    def set_version_keeper_dir(self, version_keeper_dir: str) -> None:
        import pickle
        version_path = os.path.join(version_keeper_dir, self.unit_name, 'version.pckl')
        if not os.path.exists(os.path.join(version_keeper_dir, self.unit_name)):
            return
        if not os.path.isfile(version_path):
            dump_pickle(pickle, version_path, self.unit_assumptions)
        self.assume(**load_pickle(pickle, version_path))

    def assume(self, **user_assumptions) -> None:
        """
        used by the user to assume that the version of the unit has the given assumptions
        """
        for name, value in user_assumptions.items():
            if name in self.user_assumptions and value != self.user_assumptions[name]:
                raise BadAssumption(f'Assumption {name} in {self.unit_name} already assumed to be '
                                    f'{self.user_assumptions[name]} but is now {value}.')
            self.user_assumptions[name] = value

    def check_assumption(self, **required_assumptions: tuple) -> Callable[[Callable], Callable]:
        """
        used in each function in the unit to check that the assumptions are correct
        """

        for name, value in required_assumptions.items():
            if name not in self.unit_assumptions:
                raise warnings.warn(f'Assumption {name} not found in {self.unit_name} assumptions.')
            if value != self.unit_assumptions[name]:
                raise warnings.warn(f'Assumption {name} in {self.unit_name} is {self.unit_assumptions[name]} '
                                    f'but is required to be {value}.')

        if self.SUPPRESS or self.suppressed:
            return lambda f: f

        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                for name, value in required_assumptions.items():
                    if name not in self.user_assumptions or self.user_assumptions[name] !=value:
                        raise BadAssumption(
                            f'Assumption {name} in {self.unit_name} is ' +
                            (self.user_assumptions[name] if name in self.user_assumptions else 'non existent') +
                            f' but should be {value}.')
                return func(*args, **kwargs)

            return wrapper

        return decorator
