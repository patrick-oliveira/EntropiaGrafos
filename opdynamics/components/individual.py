import random

import numpy as np

from opdynamics import max_H
from opdynamics.components.memory import (initialize_memory,
                                          probability_distribution,
                                          random_selection)
from opdynamics.math.entropy import memory_entropy
from opdynamics.math.polarity import polarity
from opdynamics.utils.types import Binary, Memory


class Individual:
    def __init__(self, kappa: float, memory_size: int):
        self.kappa = kappa
        self.memory_size = memory_size
        self.seed = random.randint(1, 100)
        self.L = initialize_memory(memory_size)
        self.L_temp = []
        self.transmissions = 0
        self.acceptances = 0
        
    @property
    def L(self):
        return self._L
    
    @property
    def X(self) -> Binary:
        """
        Return a randomly selected binary code based on the memory's probability distribution.

        Returns:
            Binary: A binary code (numpy array of bits)
        """        
        return self.select_information()
    
    @property 
    def H(self):
        return self._H
    
    @property
    def delta(self):
        return self._delta
    
    @property
    def P_array(self):
        return np.array(list(self.P.values()))
    
    @L.setter
    def L(self, memory: Memory):
        """
        Everytime the memory is updated, its probability distribution is automatically updated, so as the entropy, the values that depends on the entropy, and the polarization.

        Args:
            memory (Memory): An array of binary codes.
        """
        self._L = memory
        self.P  = probability_distribution(self.L, self.memory_size)
        self.compute_entropy()
        self.compute_polarization()

    def compute_entropy(self):
        """
        Everytime the entropy is updated, the distortion probability (delta) is automatically updated.
        """
        self._H = memory_entropy(self.P)
        self.compute_conservation_factor()
    
    def compute_conservation_factor(self):
        """
        Updates the probability of distortion due to imperfect memory..
        """        
        self._delta = 1/(np.exp(self.kappa*(max_H - self.H)/max_H) + 1)
    
    @property
    def pi(self):
        return self._pi

    def compute_polarization(self):
        self._pi = self.L[1].mean()

    def select_information(self):
        return random_selection(self.P)
    
    def update_memory(self):
        if len(self.L_temp) > 0:
            polarity_array = polarity(np.asarray(self.L_temp))
            self.L = [np.append(self.L[0], self.L_temp, axis = 0)[len(self.L_temp):], \
                      np.append(self.L[1], polarity_array, axis = 0)[len(self.L_temp):]]
            self.L_temp = []
        
    def receive_information(self, new_code: Binary) -> bool:
        if not (new_code is None):
            self.L_temp.append(new_code)
            return True
        return False
    
    def transmitted(self):
        self.transmissions += 1
        
    def received(self):
        self.acceptances += 1