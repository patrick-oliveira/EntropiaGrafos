import numpy as np
from abc import ABC, abstractmethod
from opdynamics.model.model import Model


class Statistic(ABC):
    @abstractmethod
    def compute(self, model: Model, *args, **kwargs) -> float | np.ndarray:
        ...

    @abstractmethod
    def get_rep_mean(
        self,
        statistics: np.ndarray,
        *args,
        **kwargs
    ) -> np.ndarray:
        ...
