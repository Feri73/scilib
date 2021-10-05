from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional

from base.computation import Node
from base.distributions import Distribution
from base.sets import Set

T = TypeVar('T')

Hypothesis = Distribution


class ConditionRejectedError(Exception):
    def __init__(self):
        super().__init__('Condition is not met for this hypothesis test.')


class HypothesisTestFactory(ABC, Generic[T]):
    """
    Guarantees that _power(dirac(null_hypothesis),significance)=significance
    """

    class HypothesisTest(Generic[T]):
        def __init__(self, factory: 'HypothesisTestFactory[T]', significance: float):
            self.__factory = factory
            self.__significance = significance

        @property
        def significance(self) -> float:
            return self.__significance

        def is_rejected(self, data: T) -> bool:
            return self.__factory.pvalue(data) < self.__significance

        def confidence_data_set(self) -> Set[T]:
            """
            :return: the set which, if the data is in it, the test is not rejected
            """
            return self.__factory._confidence_data_set(self.__significance)

        def power(self, alt_dist: Distribution[Hypothesis[T]]) -> float:
            """
            :return: p(is_rejected|data~hypothesis,hypothesis~alt_dist)
            """
            return self.__factory._power(alt_dist, self.__significance)

        def simulated_power(self, alt_dist: Distribution[Hypothesis[T]], n: int) -> float:
            return self.__factory._simulated_power(alt_dist, self.__significance, n)

        def required_alt_set(self, power: float) -> Set[Hypothesis[T]]:
            """
            :return: the set which, which for all h in it, power(dirac(h))>=power
            """
            return self.__factory._required_alt_set(power, self.__significance)

    def __init__(self, is_condition_rejected: Optional[Node]):
        self._is_condition_rejected = is_condition_rejected

    def make(self, significance: float) -> HypothesisTest[T]:
        return HypothesisTestFactory.HypothesisTest(self, significance)

    def pvalue(self, data: T) -> float:
        if self._is_condition_rejected is not None and self._is_condition_rejected.eval(data):
            raise ConditionRejectedError()
        return self._pvalue(data)

    @abstractmethod
    def _pvalue(self, data: T) -> float:
        pass

    @abstractmethod
    def _confidence_data_set(self, significance: float) -> Set[T]:
        pass

    def _power(self, alt_dist: Distribution[Hypothesis[T]], significance: float) -> float:
        return self._simulated_power(alt_dist, significance, 1000)

    def _simulated_power(self, alt_dist: Distribution[Hypothesis[T]], significance: float, n: int) -> float:
        test = self.make(significance)
        rejected_count = 0
        for _ in range(n):
            try:
                rejected_count += test.is_rejected(alt_dist.sample().sample())
            except ConditionRejectedError:
                pass
        return rejected_count / n

    @abstractmethod
    def _required_alt_set(self, power: float, significance: float) -> Set[Hypothesis[T]]:
        pass


HypothesisTest = HypothesisTestFactory.HypothesisTest
