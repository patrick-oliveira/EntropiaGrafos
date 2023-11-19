import random
import numpy as np

from typing import Tuple
from opdynamics import MAX_H
from opdynamics.math.entropy import memory_entropy
from opdynamics.math.polarity import polarity
from opdynamics.utils.types import Binary, Memory, Polarity
from opdynamics.model.dynamics import evaluate_information
from opdynamics.components.utils import random_selection
from opdynamics.components.memory import (initialize_memory,
                                          probability_distribution)


class Individual:
    def __init__(
        self,
        kappa: float,
        memory_size: int,
        code_length: int,
        distribution: str,
        seed: int = None,
        *args,
        **kwargs
    ):
        self.kappa = kappa
        self.memory_size = memory_size
        self.code_length = code_length
        self.seed = random.randint(1, 100) if seed is None else seed

        self.L = initialize_memory(
            memory_size,
            code_length,
            distribution,
            seed,
            *args,
            **kwargs
        )
        self.L_temp = []

        self.transmissions = 0
        self.acceptances = 0

    @property
    def L(self) -> Tuple[Memory, Polarity]:
        """
        Returns the indiviudal's memory.

        Returns:
            Tuple[Memory, Polarity]: A tuple containing the memory and
            its polarity
        """
        return self._L

    @L.setter
    def L(self, mem_pol: Tuple[Memory, Polarity]):
        """
        Everytime the memory is updated, its probability distribution
        is automatically updated, so as the entropy, the values that
        depends on the entropy, and the polarization.

        Args:
            memory (Tuple[Memory, Polarity]): An array of binary codes and
            its polarity.
        """
        self._L = mem_pol
        self.P = probability_distribution(
            self.L[0],
            self.memory_size,
            self.code_length
        )
        self.compute_entropy()
        self.compute_polarization()

    @property
    def X(self) -> Binary:
        """
        Return a randomly selected binary code based on the memory's
        probability distribution.

        Returns:
            Binary: A binary code (numpy array of bits).
        """
        return self.select_information()

    def select_information(self):
        """
        Selects a piece of information from the individual's memory with
        probability proportional to its frequency.

        Returns:
            The selected piece of information.
        """
        return random_selection(self.P)

    @property
    def H(self) -> float:
        """
        The entropy of the individual's memory.

        Returns:
            float: The entropy of the individual's memory.
        """
        return self._H

    @property
    def delta(self) -> float:
        """
        The individual's conservation factor (distortion probability).

        Returns:
            float: Probability of distortion.
        """
        return self._delta

    def compute_entropy(self):
        """
        Everytime the entropy is updated, the distortion probability (delta)
        is automatically updated.
        """
        self._H = memory_entropy(self.P)
        self.compute_conservation_factor()

    def compute_conservation_factor(self):
        """
        Updates the probability of distortion due to imperfect memory..
        """
        self._delta = 1/(np.exp(self.kappa*(MAX_H - self.H)/MAX_H) + 1)

    @property
    def pi(self):
        """
        The polarity of the individual's memory.

        Returns:
            float: The individual's polarity.
        """
        return self._pi

    def compute_polarization(self):
        """
        Computes the individual's polarization, i.e. the mean polarity of its
        memory.
        Saves it into a internal attribute.
        """
        self._pi = self.L[1].mean()

    @property
    def P_array(self):
        """
        Return the individual's memory probability distribution as an array.
        """
        return np.array(list(self.P.values()))

    def receive_information(self, new_code: Binary, p: float) -> bool:
        """
        Receives a new piece of information. Saves it into a temporary list
        to be updated into the memory later, returning a confirmation if the
        information was received or not.
        """
        new_code = evaluate_information(new_code, p)
        if new_code is not None:
            self.L_temp.append(new_code)
            return True
        return False

    def update_memory(self):
        """
        Update the individual's memory (codes and polarity) by adding the
        new information received following a FIFO policy.
        """
        if len(self.L_temp) > 0:
            polarity_array = polarity(np.asarray(self.L_temp))
            self.L = (
                np.append(
                    self.L[0],
                    self.L_temp,
                    axis=0
                )[len(self.L_temp):],
                np.append(
                    self.L[1],
                    polarity_array,
                    axis=0
                )[len(self.L_temp):])
            self.L_temp = []

    def transmitted(self):
        """
        Count a new successful transmission.
        """
        self.transmissions += 1

    def received(self):
        """
        Count a new successful reception.
        """
        self.acceptances += 1
