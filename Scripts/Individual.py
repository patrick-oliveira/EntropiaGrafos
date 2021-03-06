__package__ = None

import numpy as np
from copy import deepcopy
from Scripts.Types import Memory, Binary
from Scripts.Memory import initialize_memory, probability_distribution, random_selection
from Scripts.Entropy import empirical_entropy, max_H
from Scripts.Polarity import polarity

from multiprocessing import Pool

class Individual:
    def __init__(self, mu: int, m: int, kappa: float):
        self.kappa = kappa
        self.L = initialize_memory(mu, m)
        self.L_temp = []
        
    @property
    def L(self):
        return self._L
    
    @L.setter
    def L(self, memory: Memory):
        '''
        Everytime the memory is updated, the distribution must be updated, so as the entropy (and the values that depends on the entropy) 
        and the polarization.
        '''
        self._L = memory
        self.P  = probability_distribution(self.L)
        self.compute_entropy()
        self.compute_polarization()
    
    @property
    def mu(self):
        '''
        mu : memory size
        '''
        return len(self.L)
    
    @property
    def X(self):
        '''
        Return a randomly selected binary code, based on the frequency of codes in the memory.
        '''
        return self.select_information()
    
    # Setters are used to make it explicit that some parameters exists and are computed elsewhere, e.g. "compute_entropy".
    
    @property 
    def H(self):
        return self._H

    def compute_entropy(self):
        '''
        Everytime the entropy is updated, the conservation factor must be updated also.
        '''
        self._H = empirical_entropy(self.L, self.P)
        self.compute_conservation_factor()
    
    @property
    def delta(self):
        return self._delta
    
    def compute_conservation_factor(self):
        self._delta = 1/(np.exp(self.kappa*(max_H - self.H)/max_H) + 1)
    
    @property
    def pi(self):
        return self._pi

    def compute_polarization(self):
        # with Pool(8) as pool:
            # self._pi = sum(pool.map(polarity, self.L))/len(self.L)
        self._pi = sum(map(polarity, self.L))/len(self.L)

    def select_information(self):
        return random_selection(self.P)
    
    def update_memory(self):
        for new_code in self.L_temp:
            self.L.append(new_code)
        
        self.L = self.L[len(self.L_temp):] 
        self.L_temp = []
        
    def receive_information(self, new_code: Binary):
        if new_code != None:
            self.L_temp = self.L_temp + [new_code]