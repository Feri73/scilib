from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar('T')


def _setify_inputs(func):
    return lambda *args, **kwargs: func(*[arg if isinstance(arg, Set) else WrapperSet(arg) for arg in args],
                                        **{kw: (arg if isinstance(arg, Set) else WrapperSet(arg))
                                           for kw, arg in kwargs.items()})


class Set(ABC, Generic[T]):
    def __new__(cls, *args) -> 'Set[T]':
        if cls != Set:
            return super().__new__(cls, *args)
        assert len(args) == 1
        return super().__new__(WrapperSet, *args)

    @abstractmethod
    def __contains__(self, item: T) -> bool:
        pass

    @_setify_inputs
    def __or__(self, other: 'Set[T]') -> 'UnionSet[T]':
        return UnionSet(self, other)

    @_setify_inputs
    def __ror__(self, other: 'Set[T]') -> 'UnionSet[T]':
        return UnionSet(self, other)

    @_setify_inputs
    def __and__(self, other: 'Set[T]') -> 'IntersectionSet[T]':
        return IntersectionSet(self, other)

    @_setify_inputs
    def __rand__(self, other: 'Set[T]') -> 'IntersectionSet[T]':
        return IntersectionSet(self, other)

    @_setify_inputs
    def __invert__(self) -> 'ComplementSet[T]':
        return ComplementSet(self)


class ComplementSet(Set[T]):
    def __init__(self, set: Set[T]):
        self._set = set

    def __contains__(self, item: T) -> bool:
        return item not in self._set


class UnionSet(Set[T]):
    def __init__(self, *sets: Set[T]):
        self._sets = sets

    def __contains__(self, item: T) -> bool:
        for set in self._sets:
            if item in set:
                return True
        return False


class IntersectionSet(Set[T]):
    def __init__(self, *sets: Set[T]):
        self._sets = sets

    def __contains__(self, item: T) -> bool:
        for set in self._sets:
            if item not in set:
                return False
        return True


class WrapperSet(Set[T]):
    def __init__(self, wrapped):
        self._wrapped = wrapped

    @property
    def inside(self):
        return self._wrapped

    def __contains__(self, item: T) -> bool:
        return item in self._wrapped
