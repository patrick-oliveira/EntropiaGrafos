__package__ = None

import numpy as np
from Scripts.Types import Memory, Binary
from Scripts.Memory import initialize_memory, probability_distribution, random_selection
from Scripts.Entropy import memory_entropy
from Scripts.Parameters import max_H
import random

class Individual:
    def __init__(self, kappa: float):
        self.kappa = kappa
        self.seed = random.randint(1, 100)                    # Preciso inicializar tudo aleatoriamente, porém os resultados precisam ser reprodutíveis. Como isso deveria ser feito?
        self.L = initialize_memory()
        self.L_temp = []
        
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
    
    @L.setter
    def L(self, memory: Memory):
        """
        Everytime the memory is updated, its probability distribution is automatically updated, so as the entropy, the values that depends on the entropy, and the polarization.

        Args:
            memory (Memory): An array of binary codes.
        """
        self._L = memory
        self.P  = probability_distribution(self.L)
        self.compute_entropy()
        # self.compute_polarization()

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
    
    # @property
    # def pi(self):
    #     return self._pi

    # def compute_polarization(self):
    #     self._pi = sum([code[1] for code in self.L])/self.mu

    def select_information(self):
        return random_selection(self.P)
    
    def update_memory(self):
        # Arrume isso aqui para incluir a polaridade
        if len(self.L_temp) > 0:
            self.L = [np.append(self.L[0], self.L_temp, axis = 0)[len(self.L_temp):], self.L[1]]
            self.L_temp = []
        
    def receive_information(self, new_code: Binary):
        if not (new_code is None):
            self.L_temp.append(new_code)