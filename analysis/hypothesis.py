from typing import TypeVar
from warnings import warn

import numpy as np
import numpy.typing as npt
from statsmodels.stats.power import TTestPower

from base.computation import Func, PH, Const, Node
from base.distributions import Distribution

try:
    import scipy.stats
except ModuleNotFoundError:
    warn('scipy was not found.')

from analysis.distributions import TDistribution1D, NormalDistribution1D, IID, DiracDistribution
from analysis.sets import Interval, ProjectedSet, Integers, Numbers, PointSet
from base.hypothesis import HypothesisTestFactory, Hypothesis, HypothesisTest
from base.sets import Set

npc = Const(np)

T = TypeVar('T')


class ShapiroWilkTest(HypothesisTestFactory[npt.NDArray]):
    def __init__(self):
        super().__init__(None)

    def _pvalue(self, data: npt.NDArray) -> float:
        return scipy.stats.shapiro(data).pvalue

    def _confidence_data_set(self, significance: float) -> Set[npt.NDArray]:
        raise NotImplementedError()

    def _required_alt_set(self, power: float, significance: float) -> Set[Hypothesis[npt.NDArray]]:
        raise NotImplementedError()


class TTest1Sample1D(HypothesisTestFactory[npt.NDArray]):
    class ConfidenceDataSet(ProjectedSet[npt.NDArray]):
        def __init__(self, test: 'TTest1Sample1D', significance: float,
                     sample_n: Set[int] = Integers & Interval(2, np.inf), sample_mean: Set[float] = Numbers,
                     sample_std: Set[float] = Numbers):
            super().__init__(sample_n=sample_n, sample_mean=sample_mean, sample_std=sample_std)
            self.__test = test
            self.__significance = significance

        def __make_tdist(self, sample_n: int, sample_std: float):
            return TDistribution1D(sample_n - 1, self.__test.null_hypothesis_mean,
                                   sample_std / np.sqrt(sample_n)).distribution

        def __get_interval(self, tdist) -> Interval:
            if self.__test.alt_direction == 'two-sided':
                lower, upper = tdist.interval(1 - self.__significance)
            elif self.__test.alt_direction == 'less':
                upper = tdist.interval(1 - self.__significance * 2)[0]
                lower = -np.inf
            elif self.__test.alt_direction == 'greater':
                lower = tdist.interval(1 - self.__significance * 2)[1]
                upper = np.inf
            return Interval(lower, upper)

        def _solve(self, sample_n: PointSet[int] = None, sample_mean: PointSet[float] = None,
                   sample_std: PointSet[float] = None) -> 'TTest1Sample1D.ConfidenceDataSet':
            if (sample_mean is None) + (sample_n is None) + (sample_std is None) != 1:
                raise NotImplementedError('Can only solve for one unknown.')

            if sample_mean is None:
                tdist = self.__make_tdist(sample_n.element, sample_std.element)
                return TTest1Sample1D.ConfidenceDataSet(self.__test, self.__significance,
                                                        sample_n, self.__get_interval(tdist), sample_std)
            if sample_n is None:
                n = 2
                while True:
                    tdist = self.__make_tdist(n, sample_std.element)
                    if sample_mean.element in self.__get_interval(tdist):
                        return TTest1Sample1D.ConfidenceDataSet(self.__test, self.__significance,
                                                                Integers & Interval(1, n), self.__get_interval(tdist),
                                                                sample_std)
            else:
                raise NotImplementedError()

        def __contains__(self, item: npt.NDArray) -> bool:
            return not self.__test.make(self.__significance).is_rejected(item)

    class RequiredAltSet(ProjectedSet[Distribution[npt.NDArray]]):
        def __init__(self, test: 'TTest1Sample1D', significance: float, power: float,
                     sample_n: Set[int] = Integers & Interval(2, np.inf),
                     abs_pop_mean: Set[float] = Interval(0, np.inf), pop_std: Set[float] = Numbers):
            super().__init__(sample_n=sample_n, abs_pop_mean=abs_pop_mean, pop_std=pop_std)
            self.__test = test
            self.__significance = significance
            self.__power = power

        def __stats_alt(self) -> str:
            return self.__test.alt_direction if self.__test.alt_direction == 'two-sided' else 'one-sided'

        def _solve(self, sample_n: PointSet[int] = None, abs_pop_mean: PointSet[float] = None,
                   pop_std: PointSet[float] = None) -> 'TTest1Sample1D.RequiredAltSet':
            if (abs_pop_mean is None) + (sample_n is None) + (pop_std is None) != 1:
                raise NotImplementedError('Can only solve for one unknown.')
            if abs_pop_mean is None or pop_std is None:
                st_mean = TTestPower().solve_power(None, sample_n.element, self.__significance, self.__power,
                                                   self.__stats_alt())
                if abs_pop_mean is None:
                    abs_pop_mean = Interval(st_mean * pop_std.element, np.inf)
                else:
                    pop_std = Interval(0, abs_pop_mean.element / st_mean)
            if sample_n is None:
                sample_n = Integers & Interval(
                    TTestPower().solve_power(abs_pop_mean.element / pop_std.element, None, self.__significance,
                                             self.__power, self.__stats_alt()),
                    np.inf)
            return TTest1Sample1D.RequiredAltSet(self.__test, self.__significance, self.__power,
                                                 sample_n, abs_pop_mean, pop_std)

        def __contains__(self, item: Distribution[npt.NDArray]) -> bool:
            return self.__test.make(self.__significance).power(DiracDistribution(item)) >= self.__power

    def __init__(self, condition_test: HypothesisTest = ShapiroWilkTest().make(.001), alt_direction: str = 'two-sided'):
        super().__init__(Func((lambda x: False) if condition_test is None else condition_test.is_rejected)(PH('data')) |
                         (((npc.mean(PH('data')) < 0) & (alt_direction == 'greater')) |
                          ((npc.mean(PH('data').forget) > 0) & (alt_direction == 'less'))))
        self.__alt_direction = alt_direction

    @property
    def alt_direction(self):
        return self.__alt_direction

    def _pvalue(self, data: npt.NDArray) -> float:
        return scipy.stats.ttest_1samp(data, 0, nan_policy='raise', alternative=self.alt_direction).pvalue

    def _power(self, alt_dist: Distribution[Distribution[npt.NDArray]], significance: float) -> float:
        if isinstance(alt_dist, DiracDistribution) and isinstance(alt_dist.value, IID) and \
                isinstance(alt_dist.value.internal, NormalDistribution1D) and \
                isinstance(alt_dist.value.n_dist, DiracDistribution):
            if self.alt_direction == 'greater' and alt_dist.value.internal.mean < 0 or \
                    self.alt_direction == 'less' and alt_dist.value.internal.mean > 0:
                raise ValueError()
            return TTestPower().solve_power(abs(alt_dist.value.internal.mean / alt_dist.value.internal.std),
                                            alt_dist.value.n_dist.value, significance, None,
                                            self.alt_direction if self.alt_direction == 'two-sided' else 'one-sided')
        else:
            super(TTest1Sample1D, self)._power(alt_dist, significance)

    def _confidence_data_set(self, significance: float) -> 'TTest1Sample1D.ConfidenceDataSet':
        return TTest1Sample1D.ConfidenceDataSet(self, significance)

    def _required_alt_set(self, power: float, significance: float) -> 'TTest1Sample1D.RequiredAltSet':
        return TTest1Sample1D.RequiredAltSet(self, significance, power)


class PermutationTest(HypothesisTestFactory[T]):
    def __init__(self, null_hypothesis: Distribution[T], is_trial_rejected: Node, n_trials: int):
        super().__init__(None)

        self.__null_hypothesis = null_hypothesis
        self.__is_trial_rejected = is_trial_rejected
        self.__n_trials = n_trials

    def _pvalue(self, data: T) -> float:
        is_trial_rejected_count = 0
        for _ in range(self.__n_trials):
            is_trial_rejected_count += self.__is_trial_rejected.eval(data, self.__null_hypothesis.sample())
        return is_trial_rejected_count / self.__n_trials

    def _confidence_data_set(self, significance: float) -> Set[T]:
        raise NotImplementedError()

    def _required_alt_set(self, power: float, significance: float) -> Set[Hypothesis[T]]:
        raise NotImplementedError()
