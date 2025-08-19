import os
from abc import ABC, abstractmethod


class PathInferrer(ABC):
    @abstractmethod
    def infer(self, path: str) -> str:
        pass


class MacroPathInferrer(PathInferrer):
    def __init__(self, os: str = None):
        self.__macros = {}
        if os is None:
            import os as os_module
            os = os_module.name
        self.__os = os

    def add_macro(self, macro: str, value: str) -> 'MacroPathInferrer':
        self.__macros[macro] = value
        return self

    def infer(self, path: str) -> str:
        for macro, value in self.__macros.items():
            path = path.replace(macro, value)
        if self.__os == 'posix':
            path = path.replace('\\', '/')
        return path


class Combined(PathInferrer):
    def __init__(self, *inferrers):
        self.__inferrers = inferrers

    def infer(self, path: str) -> str:
        for inferrer in self.__inferrers:
            path = inferrer.infer(path)
        return path


INFERRER = MacroPathInferrer()


class UniversalPath:
    def __init__(self, path: str, inferer: PathInferrer = None):
        self.__path = path
        self.__inferer = inferer
        self.resolve = self.infer

    def with_inferrer(self, inferrer: PathInferrer) -> 'UniversalPath':
        return UniversalPath(self.__path, inferrer)

    def infer(self) -> str:
        inferrer = self.__inferer
        if inferrer is None:
            inferrer = globals()['INFERRER']
        return inferrer.infer(self.__path)

    def __str__(self):
        return self.infer()

    def __fspath__(self):
        return str(self)

    def __truediv__(self, other):
        return UniversalPath(self.__path + os.path.sep + other)

    def __eq__(self, other):
        if isinstance(other, UniversalPath):
            return str(self) == str(other)
        return False

    def __repr__(self):
        return f'UniversalPath({self})'
