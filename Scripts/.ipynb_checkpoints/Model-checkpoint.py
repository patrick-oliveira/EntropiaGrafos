zimport networkx as nx
import numpy as np
from random import sample

from Scripts.Types import Graph, TransitionProbabilities
from Scripts.Individual import Individual
from Scripts.ModelDynamics import acceptance_probability, _indInfo, _indTendency, get_transition_probability
from Scripts.Entropy import JSD

from time import time

class Model:
    def __init__(self, N: int, pa: int,            # Graph Parameters
                       mu: int, m: int,            # Memory Parameters
                       kappa: float, lambd: float, # Proccess Parameters
                       alpha: float, omega: float, # Population Parameters
                       gamma: float,               # Information Dissemination Parameters
                       seed: int = 42):
        self.N  = N
        self.pa = pa
        
        self.mu = mu
        self.m  = m
        self.kappa = kappa
        self.alpha = alpha
        self.omega = omega
        self.lambd = lambd
        self.gamma = gamma
        
        self._seed = seed
        
        self.build_model()

    @property
    def H(self):
        return self._H
    
    @property
    def G(self): 
        return self._G # Defined at "build_model()"
        
    @property
    def seed(self):
        return self._seed
        
    def build_model(self):
        self._G = nx.barabasi_albert_graph(self.N, self.pa, self.seed)
        self.initialize_nodes()
        self.compute_entropy()
        self.compute_polarity()
        self.group_individuals()
        self.compute_polarity_chance()
        self.define_distortion_probabilities()
        self.compute_edge_weights()
        self.compute_sigma_attribute()
        
    def update_model(self):
        for node in self.G:
            self.G.nodes[node]['Object'].update_memory()
            
        self.compute_entropy()
        self.compute_polarity()
        self.compute_polarity_chance()
        self.define_distortion_probabilities()
        
    def initialize_nodes(self):
        '''
        Attribute to each node a new 'Individual' object.
        '''
        nx.set_node_attributes(self.G, {node:Individual(self.mu, self.m, self.kappa) for node in self.G}, name = 'Object')
            
    def group_individuals(self):
        indices = sample(list(range(self.N)), k = self.N)
        group_a = indices[:int(self.alpha*self.N)]
        group_b = indices[int(self.alpha*self.N): int(self.alpha*self.N) + int(self.omega*self.N)]
        group_c = indices[int(self.alpha*self.N) + int(self.omega*self.N):]
        
        attribute_dict = {}
        attribute_dict.update({list(self.G.nodes())[i]:'Up' for i in group_a})
        attribute_dict.update({list(self.G.nodes())[i]:'Down' for i in group_b})
        attribute_dict.update({list(self.G.nodes())[i]:'-' for i in group_c})
        
        nx.set_node_attributes(self.G, attribute_dict, name = 'Tendency')
        
    def compute_polarity_chance(self):
        for node in self.G:
            node_object = self.indInfo(node)
            mean_polarity_neighbours = np.mean([self.indInfo(neighbor).pi for neighbor in self.G.neighbors(node)])
            setattr(node_object, 'xi', self.lambd*abs(node_object.pi - mean_polarity_neighbours))
            
    def define_distortion_probabilities(self):
        for node in self.G:
            individual = self.indInfo(node)
            tendency   = self.indTendency(node)
            setattr(individual, 'DistortionProbability', get_transition_probability(individual, tendency))
            
    def compute_edge_weights(self):
        nx.set_edge_attributes(self.G, {(u, v):(1 - JSD(self.indInfo(u).P, self.indInfo(v).P)) for u, v in self.G.edges}, 'Distance')
        
    def compute_sigma_attribute(self):
        for node in self.G:
            global_proximity = sum([self.G.edges[(node, neighbor)]['Distance'] for neighbor in self.G.neighbors(node)])
            setattr(self.indInfo(node), 'sigma', global_proximity)
            
    def get_acceptance_probability(self, u: int, v:int) -> float:
        return acceptance_probability(self.G, u, v, self.gamma)
        
    def compute_entropy(self):
        self._H = sum([self.indInfo(node).H for node in self.G])/self.N
        
    def compute_polarity(self):
        self.pi = sum([self.indInfo(node).pi for node in self.G])/self.N
        
    def indInfo(self, node: int) -> Individual:
        return _indInfo(self.G, node)
    
    def indTendency(self, node: int) -> Individual:
        return _indTendency(self.G, node) 
        
        
        


