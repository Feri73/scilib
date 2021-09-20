from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Collection, Union, Optional, Type

from base.computation import Variable, Node, Function

T = TypeVar('T')


class Distribution(ABC, Generic[T]):
    __is_registered: bool = False

    def __init__(self):
        if not self.__is_registered:
            Function.register(self.variable_type_creator)

    @abstractmethod
    def sample(self, n: int = None) -> Union[T, Collection[T]]:
        pass

    def pdf(self, data: T) -> float:
        raise NotImplementedError()

    def cdf(self, data: T) -> float:
        raise NotImplementedError()

    @property
    def var(self) -> 'VariableDistribution':
        return CopyVariableDistribution(self)

    @staticmethod
    def variable_type_creator(*args, **kwargs) -> Optional[Type['WrapVariableDistribution[T]']]:
        found = False
        for arg in list(args) + list(kwargs.values()):
            if isinstance(arg, Node) and arg.is_active and not isinstance(arg, VariableDistribution):
                return None
            if isinstance(arg, VariableDistribution) and arg.is_active:
                found = True
        return WrapVariableDistribution if found else None


class VariableDistribution(Variable, Distribution[T], ABC):
    def __init__(self):
        Variable.__init__(self, self.sample)


class CopyVariableDistribution(VariableDistribution[T]):
    def __init__(self, orig_dist: Distribution):
        super().__init__()
        self.__orig_dist = orig_dist

    @property
    def orig(self) -> Distribution:
        return self.__orig_dist

    def sample(self, n: int = None) -> Union[T, Collection[T]]:
        return self.orig.sample(n)

    def pdf(self, data: T) -> float:
        return self.orig.pdf(data)

    def cdf(self, data: T) -> float:
        return self.orig.cdf(data)


class WrapVariableDistribution(VariableDistribution[T]):
    def __init__(self, variable: Variable):
        super().__init__()
        self.__variable = variable

    def sample(self, n: int = None) -> Union[T, Collection[T]]:
        if n is None:
            return self.__variable.eval()
        else:
            return [self.__variable.eval() for _ in range(n)]
