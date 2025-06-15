import random
import numpy as np

from numpy.typing import NDArray
from typing import List
from opdynamics.math.entropy import memory_entropy
from opdynamics.utils.types import Memory
from opdynamics.components.memory import (
    initialize_memory,
    probability_distribution,
    polarity
)


class Individual:
    def __init__(
        self,
        kappa: float,
        memory_size: int,
        distribution: str = "multivariate_normal",
        info_dimension: int = 2,
        tendency: int = 0,
        **kwargs
    ):
        self.kappa = kappa
        self.memory_size = memory_size
        self.tendency = tendency
        # Could this be improved for reproducibility sake?
        self.seed = random.randint(1, 100)

        self.L = initialize_memory(  # type: ignore
            memory_size = memory_size,
            info_dimension = info_dimension,
            distribution = distribution,
            **kwargs
        )
        self.L_temp: List[NDArray] = []

        self.stats = {
            "transmissions": 0,
            "acceptances": 0
        }

    @property
    def L(self):
        return self._L

    @property
    def X(self) -> np.ndarray:
        return self.select_information()

    @property
    def H(self):
        return self._H

    @property
    def pi(self):
        return self._pi

    @property
    def delta(self):
        return self._delta

    @L.setter  # type: ignore
    def L(self, memory: Memory):
        """
        Everytime the memory is updated, its probability distribution is
        automatically updated, so as the entropy, the values that depends on
        the entropy, and the polarization.

        Args:
            memory (Memory): An array of binary codes.
        """
        self._L = memory
        self.compute_entropy()
        self.compute_polarization()

    def compute_entropy(self):
        """
        Everytime the entropy is updated, the distortion probability (delta)
        is automatically updated.
        """
        self._H = memory_entropy(self.L)
        self.compute_conservation_factor()

    def compute_conservation_factor(self):
        """
        Updates the probability of distortion due to imperfect memory..
        """
        self._delta = 1 / (np.exp(self.kappa * (1 / self.H)) + 1)

    def compute_polarization(self):
        self._pi = self.L["polarities"].mean()

    def select_information(self):
        return self.L["distribution"].sample()

    def update_memory(self):
        if len(self.L_temp) > 0:
            L_temp = np.stack(self.L_temp, axis = 0)
            new_codes = np.concatenate([self.L["codes"], L_temp], axis = 0) # noqa
            new_codes = new_codes[len(self.L_temp):]

            new_polarities = polarity(new_codes)
            new_distribution = probability_distribution(new_codes)

            self.L = Memory(
                codes = new_codes,
                polarities = new_polarities,
                distribution = new_distribution
            )

            self.L_temp = []

    def receive_information(self, new_code: np.ndarray) -> bool:
        if not (new_code is None):
            self.L_temp.append(new_code.squeeze(0))
            return True
        return False

    def transmitted(self):
        self.stats["transmissions"] += 1

    def received(self):
        self.stats["acceptances"] += 1
