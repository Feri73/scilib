from abc import ABC, abstractmethod
from typing import TypeVar, Any, Dict, Tuple, Union

from base.computation import PH, Func, Const, Node
from base.sets import Set

T = TypeVar('T')
FuncArgs = Tuple[Tuple, Dict[str, Any]]


class PointSet(Set[T]):
    def __init__(self, element: T):
        self.__element = element

    @property
    def element(self):
        return self.__element

    def __contains__(self, item: T) -> bool:
        return item == self.element


class Interval(Set[T]):
    def __init__(self, lower: T, upper: T, predicate: Node = None):
        self.__lower = lower
        self.__upper = upper
        self.__has_pred = predicate is not None
        self.__pred = Func(lambda x: True)(PH()) if predicate is None else predicate

    @property
    def lower(self) -> T:
        return self.__lower

    @property
    def upper(self) -> T:
        return self.__upper

    @property
    def has_predicate(self) -> bool:
        return self.__has_pred

    def where(self, predicate: Node) -> 'Interval[T]':
        return Interval(self.lower, self.upper, self.__pred.f(PH('*')) & predicate.f(PH('*').forget))

    def __contains__(self, item: T) -> bool:
        return self.__pred.eval(item) and self.lower <= item <= self.upper

    def __and__(self, other: Set[T]) -> Set[T]:
        if isinstance(other, Interval):
            return Interval(max(self.lower, other.lower), min(self.upper, other.upper), self.__pred).where(other.__pred)
        else:
            return super().__and__(other)


class ProjectedSet(Set[T], ABC):
    def __init__(self, **kwargs: Union[Set[Any], Any]):
        """
        :param kwargs: Specifies, for each keyword, a set that the values of the associated parameter is a subset of
        """
        self._kwargs = kwargs
        for kw, arg in kwargs.items():
            if not isinstance(arg, Set):
                arg = PointSet(arg)
            setattr(self, kw, property(lambda arg=arg: arg))

    def solve(self, **kwargs: Union[Set[Any], Any]) -> 'ProjectedSet':
        """
        :param kwargs: Specifies, for each keyword, a set that the values of the associated parameter is a subset of
        in the returned ProjectedSet
        :return: The ProjectedSet in which, for each keyword in kwargs, the value is the same as that of kwargs,
        and for any keyword in self._kwargs that is not in kwargs, its value is the minimum set that the values
         of the associated parameter is a subset of
        """
        for kw, arg in kwargs.items():
            if not isinstance(arg, Set):
                kwargs[kw] = PointSet(arg)
        return self._solve(**kwargs)

    @abstractmethod
    def _solve(self, **kwargs: Set[Any]) -> 'ProjectedSet':
        pass

    def __contains(self, **kwargs: Union[Set[Any], Any]) -> bool:
        for kw in list(kwargs.keys()):
            val = kwargs.pop(kw)
            if val in getattr(self.solve(**kwargs), kw):
                return True
            kwargs[kw] = val
        return False


Empty = Set([])

try:
    import numpy as np

    Integers = Interval(-np.inf, np.inf, Const(np).array(PH()).dtype == np.int)
    Numbers = Interval(-np.inf, np.inf)
except ModuleNotFoundError:
    pass
