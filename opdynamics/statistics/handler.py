import numpy as np
import pickle

from copy import deepcopy
from typing import List, Dict
from opdynamics.model.model import Model
from opdynamics.statistics import STATISTICS
from opdynamics.statistics.abstract_statistic import Statistic


class StatisticHandler:
    def __init__(self, statistics: List[str], *args, **kwargs):
        self.statistics: Dict[str, Statistic] = {}
        self.statistics_values: Dict[str, List[float | np.ndarray]] = {}
        self.repetitions: Dict[int, Dict[str, List[float | np.ndarray]]] = {}
        self.current_rep = 0

        for statistic in statistics:
            self._new_statistic(
                statistic,
                STATISTICS[statistic](*args, **kwargs)
            )

    def _new_statistic(self, name: str, statistic: Statistic):
        if name not in self.statistics.keys():
            self.statistics[name] = statistic
            self.statistics_values[name] = []

    def update_statistics(self, model: Model) -> None:
        for statistic in self.statistics.keys():
            new_measure = self.statistics[statistic].compute(model)
            self.statistics_values[statistic].append(new_measure)

    def reset_statistics(self, hard_reset: bool = False) -> None:
        if hard_reset:
            self.statistics = {}
            self.statistics_values = {}
        else:
            for statistic in self.statistics.keys():
                self.statistics_values[statistic] = []

    def end_repetition(self) -> None:
        # take a snapshot and increase the repetition counter
        self.repetitions[self.current_rep] = deepcopy(self.statistics_values)
        self.current_rep += 1

        # clean statistics for next repetition
        self.reset_statistics()

    def mean_along_repetitions(self) -> Dict[str, np.ndarray]:
        assert len(self.repetitions) > 0, 'At least one repetition should be '\
                            'completed to extract mean values of statistics.'

        mean_statistics = {}
        rep_stats = self.repetitions.values()
        for st in self.statistics.keys():
            # get the rep mesures for the statistic and convert to array
            stats_array = [measures[st] for measures in rep_stats]
            stats_array = np.asarray(stats_array)
            # compute the mean of the statistic along repetitions
            mean_statistics[st] = self.statistics[st].get_rep_mean(
                stats_array
            )

        return mean_statistics

    def load_from_files(self, files: List[str]):
        for k, f in enumerate(files):
            self.repetitions[k] = pickle.load(open(f, "rb"))
        self.current_rep = len(files)
