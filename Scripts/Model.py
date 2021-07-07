import networkx as nx
import numpy as np
from random import sample

from Scripts.Types import Dict
from Scripts.Individual import Individual
from Scripts.ModelDynamics import acceptance_probability, _indInfo, _indTendency, get_transition_probability, evaluate_information, distort
from Scripts.Entropy import JSD
from Scripts.Parameters import pa, mu, m


from time import time

class Model:
    def __init__(self, N: int, pa: int,            # Graph Parameters
                       mu: int, m: int,            # Memory Parameters
                       kappa: float, lambd: float, # Proccess Parameters
                       alpha: float, omega: float, # Population Parameters
                       gamma: float,               # Information Dissemination Parameters
                       seed: int = 42,
                       initialize: bool = True):
        self.N  = N
        self.pa = pa
        
        self.mu = mu
        self.m  = m
        self.kappa = kappa
        self.alpha = alpha
        self.omega = omega
        self.lambd = lambd
        self.gamma = gamma
        
        self.seed = seed
        
        
        self.create_graph()
        if initialize: self.initialize_model_info()

    @property
    def H(self):
        return self._H
    
    @property
    def G(self): 
        return self._G # Defined at "build_model()"
        
    def create_graph(self):
        self._G = nx.barabasi_albert_graph(self.N, self.pa, self.seed)
        
    def initialize_model_info(self):
        self.initialize_nodes()
        self.group_individuals()
        
        for node in self.G:
            self.update_node_info(node)
            
        # self.compute_polarity_chance() #
        # self.define_distortion_probabilities() #
        self.compute_edge_weights()
        self.compute_sigma_attribute() #
        
        self.compute_graph_entropy()
        self.compute_graph_polarity()
        
    def update_model(self):
        for node in self.G:
            self.update_node_info(node, update_memory = True)
            
        # self.update_memory() #
        # self.compute_polarity_chance() # 
        # self.define_distortion_probabilities() #
        
        self.compute_graph_entropy()
        self.compute_graph_polarity()
        
        
    def update_node_info(self, node: int, update_memory: bool = False):
        individual = self.indInfo(node)
        tendency   = self.indTendency(node)
        
        if update_memory:
            individual.update_memory()
        
        # compute polarity chance()
        mean_polarity_neighbours = np.mean([self.indInfo(neighbor).pi for neighbor in self.G.neighbors(node)])
        setattr(self.indInfo(node), 'xi', self.lambd*abs(individual.pi - mean_polarity_neighbours))       
        
         # define distortion probabilities
        setattr(individual, 'DistortionProbability', get_transition_probability(individual, tendency))
        
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
        
    # def update_memory(self):
    #     for node in self.G:
    #         self.indInfo(node).update_memory()
        
    # def compute_polarity_chance(self):
    #     for node in self.G:
    #         node_object = self.indInfo(node)
    #         mean_polarity_neighbours = np.mean([self.indInfo(neighbor).pi for neighbor in self.G.neighbors(node)])
    #         setattr(node_object, 'xi', self.lambd*abs(node_object.pi - mean_polarity_neighbours))
            
    # def define_distortion_probabilities(self):
    #     for node in self.G:
    #         individual = self.indInfo(node)
    #         tendency   = self.indTendency(node)
    #         setattr(individual, 'DistortionProbability', get_transition_probability(individual, tendency))
        
    def compute_sigma_attribute(self):
        for node in self.G:
            global_proximity = sum([self.G.edges[(node, neighbor)]['Distance'] for neighbor in self.G.neighbors(node)])
            setattr(self.indInfo(node), 'sigma', global_proximity)
            
    def compute_edge_weights(self):
        nx.set_edge_attributes(self.G, {(u, v):(1 - JSD(self.indInfo(u).P, self.indInfo(v).P)) for u, v in self.G.edges}, 'Distance')
            
    def get_acceptance_probability(self, u: int, v:int) -> float:
        return acceptance_probability(self.G, u, v, self.gamma)
        
    def indInfo(self, node: int) -> Individual:
        return _indInfo(self.G, node)
    
    def indTendency(self, node: int) -> Individual:
        return _indTendency(self.G, node) 
        
    def compute_graph_entropy(self):
        self._H = sum([self.indInfo(node).H for node in self.G])/self.N
        
    def compute_graph_polarity(self):
        self.pi = sum([self.indInfo(node).pi for node in self.G])/self.N
        
    def compute_info_distribution(self):
        hist = {info:None for info in self.indInfo(0).P.keys()}
        N = 0
        
        for node in self.G:
            for info in self.indInfo(node).L:
                hist[info] += 1
                N += 1
        
        dist = {}
        
        for info in hist.keys():
            dist[info] = hist[info]/N
        
        return hist, dist
    

def evaluateModel(T: int,
                  kappa: float, lambd: float,
                  alpha: float, omega: float,
                  gamma: float,
                  seed: [int] = 42):
    '''
    Input:
    
    Evaluate a new model over T iterations.
    '''
    print("Initializing model with parameters")
    print("N = {} - pa = {} - mu = {} - m = {} - kappa = {} - lambda = {} - \
          alpha = {} - omega = {} - gamma = {}".format(N, pa, mu, m, kappa, lambd, alpha, omega, gamma))
    
    model = Model(N, pa, mu, m, kappa, lambd, alpha, omega, gamma, seed)
    
    statistics = {}
    statistics['H - seed = {}'.format(model.seed)]  = []
    statistics['pi - seed = {}'.format(model.seed)] = []
    update_statistics(model, statistics)

    elapsedTime = 0

    for i in range(T):
        execution_time = simulate(model)
        elapsedTime += execution_time
        # print("Iteration {} ended - Execution Time = {}".format(i, execution_time))
        update_statistics(model, statistics)
        
    return elapsedTime, statistics

def simulate(M):
    '''
    Input:
        M: A population model.
        
    Execute one iteration of the dissemination model, updating the model's parameters at the end. Return the execution time (minutes).
    '''
    
    
    start = time()
    for u, v in M.G.edges():
        u_ind = M.indInfo(u)
        v_ind = M.indInfo(v)
        u_ind.receive_information(evaluate_information(distort(v_ind.X, v_ind.DistortionProbability), M.get_acceptance_probability(u, v)))
        v_ind.receive_information(evaluate_information(distort(u_ind.X, u_ind.DistortionProbability), M.get_acceptance_probability(v, u)))     
    end = time()
    
    M.update_model()
    return (end - start)/60

def update_statistics(M, statistics: Dict):
    '''
    Input:
        M: A population model.
        statistics: A dictionary to accumulate statistics computed from "M"
    '''
    statistics['H - seed = {}'.format(M.seed)].append(M.H)
    statistics['pi - seed = {}'.format(M.seed)].append(M.pi)