from typing import Union, TypeVar, Collection, Generic

import numpy as np
import numpy.typing as npt
import scipy.stats

from base.distributions import Distribution

T = TypeVar('T')
D = TypeVar('D', bound=Distribution)


class ScipyDistribution1D(Distribution[float]):
    def __init__(self, scipy_dist):
        super().__init__()
        self._dist = scipy_dist

    @property
    def distribution(self):
        return self._dist

    def sample(self, n: int = None) -> Union[float, npt.NDArray]:
        return self._dist.rvs(n)

    def pdf(self, data: float) -> float:
        return self._dist.pdf(data)

    def cdf(self, data: float) -> float:
        return self._dist.cdf(data)


class NormalDistribution1D(ScipyDistribution1D):
    def __init__(self, mean: float, std: float):
        super().__init__(scipy.stats.norm(mean, std))

    @property
    def mean(self) -> float:
        return self._dist.mean()

    @property
    def std(self) -> float:
        return self._dist.std()


class TDistribution1D(ScipyDistribution1D):
    def __init__(self, df: int, mean: float, std: float):
        super().__init__(scipy.stats.t(df, mean, std))

    @property
    def df(self):
        return self._dist.df()

    @property
    def mean(self) -> float:
        return self._dist.mean()

    @property
    def std(self) -> float:
        return self._dist.std()


class DiracDistribution(Distribution[T]):
    def __init__(self, value: T):
        super().__init__()
        self.__value = value

    @property
    def value(self) -> T:
        return self.__value

    def sample(self, n: int = None) -> Union[T, Collection[T]]:
        return self.value if n is None else [self.value] * n

    def pdf(self, data: T) -> float:
        return np.inf if data == self.value else 0.

    def cdf(self, data: T) -> float:
        return float(data >= self.value)


class IID(Generic[D, T], Distribution[Collection[T]]):
    def __init__(self, internal: D, n_dist: Distribution[int]):
        super().__init__()
        self.__internal = internal
        self.__n_dist = n_dist

    @property
    def internal(self) -> D:
        return self.__internal

    @property
    def n_dist(self) -> Distribution[int]:
        return self.__n_dist

    def sample(self, n: int = None) -> Union[Collection[T], Collection[Collection[T]]]:
        if n is None:
            return self.internal.sample(self.n_dist.sample())
        else:
            return [self.internal.sample(self.n_dist.sample()) for _ in range(n)]

    def pdf(self, data: Collection[T]) -> float:
        return self.n_dist.pdf(len(data)) * np.prod([self.internal.pdf(d) for d in data])
