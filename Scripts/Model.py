import networkx as nx
import numpy as np
from random import sample

from Scripts.Types import Dict
from Scripts.Individual import Individual
from Scripts.ModelDynamics import acceptance_probability, get_transition_probabilities, evaluate_information, distort, getInfo, getTendency
from Scripts.Entropy import JSD
from Scripts.Parameters import pa, memory_size, code_length, seed, N
from time import time

class Model:
    def __init__(self, N: int, pa: int,            # Graph Parameters
                       mu: int, m: int,            # Memory Parameters
                       kappa: float, lambd: float, # Proccess Parameters
                       alpha: float, omega: float, # Population Parameters
                       gamma: float,               # Information Dissemination Parameters
                       initialize: bool = True):
        """
        
        The model is created by the following sequence of steps:
            1. The underlying network is built. Until now, the only network used is the Barabasi-Albert, but it's possible to generalize the model for different network topologies.
            2. The function "initialize_nodes" is called and an 'Individual" object is instantiated and attributed to each node in the network.
            3. The individuals are grouped in 3 different groups, one for individuals who polarize upwards, another for individuals who polarize downwards, and another for neutral individuals. The proportion of individuals for each group is defined by the parameters alpha and omega.
            4. The function "update_node_info" is called to compute polarization probabilities. The same function is used to update the model after each iteration, therefore there is a loop calling the "update_memory" function for each node in the network.
            5. The function "compute_edge_weight" is called to attribute weights to each edge using the Jensen-Shannon Divergence.
            6. The function "compute_sigma_attribute" is called to measure the popularity of each node based on the connection weight with each neighbour.
            7. The function "compute_graph_entropy" is called to calculate the mean entropy of the network.
            8. The function "compute_graph_polarity" is called to calculate the mean polarity of the network.
            
        The steps 4 to 8 are repeated at each iteration to update the model's values.

        Args:
            N (int): Number of nodes in the network.
            pa (int): 'Preferential Attachment' parameters (exclusive to the Barabasi-Albert network; the choice of network topology can be generalized later)
            mu (int): Memory size.
            m (int): Length of binary codes.
            kappa (float): Conservation factor.
            lambd (float): Polarization coefficient.
            alpha (float): Proportion of individuals who polarize upwards.
            omega (float): Proportion of individuals who polarize downwards.
        """        
        # Network parameters
        self.N  = N
        self.pa = pa
        
        # Individual parameters
        self.mu    = mu
        self.m     = m
        self.kappa = kappa
        self.alpha = alpha
        self.omega = omega
        self.lambd = lambd
        self.gamma = gamma
        
        self.seed = seed
        
        
        self.create_graph()
        if initialize: self.initialize_model_info()

    @property
    def H(self) -> float:
        """
        Mean Entropy

        Returns:
            [float]: Mean entropy of all the individuals
        """        
        return self._H
    
    @property
    def pi(self) -> float:
        """
        Mean polarity.

        Returns:
            float: Mean polarity value of all individuals.
        """        
        return self._pi
    
    @property
    def G(self): 
        return self._G # Defined at "build_model()"
        
    def create_graph(self):
        """
        Edite esta função para permitir a criação de redes com topologias diferentes. A função deve receber um nome identificando o tipo de rede, e/ou uma função de criação da rede.
        """        
        self._G = nx.barabasi_albert_graph(self.N, self.pa, self.seed)
        
    def initialize_model_info(self):
        """
        Initialize the model by creating the individual's information and computing the necessary parameters and measures.
        """        
        self.initialize_nodes()
        self.group_individuals()
        
        for node in self.G:
            self.update_node_info(node)
            
        self.compute_edge_weights()
        self.compute_sigma_attribute()
        self.compute_graph_entropy()
        # self.compute_graph_polarity()
        
    def update_model(self):
        """
        Updates the network information (individual's information and all measures based on entropy) after each iteration.
        """        
        for node in self.G:
            self.update_node_info(node, update_memory = True)
        
        self.compute_edge_weights()
        self.compute_sigma_attribute()
        self.compute_graph_entropy()
        # self.compute_graph_polarity()
        
        
    def update_node_info(self, node: int, update_memory: bool = False):
        """
        A function used to update the state of each node in the network. 
        After each iteration, this function is called with update_memory = True to possibly include new information in each individual's memory. After that, polarization and distortion probabilities are updated.

        Args:
            node (int): A vertex id.
            update_memory (bool, optional): A boolean which specifies if the individual's memory must be updated or not. Defaults to False.
        """        
        individual = self.indInfo(node)
        tendency   = self.indTendency(node)
        
        if update_memory:
            individual.update_memory()
            
        # compute polarity probability
        # mean_polarity_neighbours = np.mean([self.indInfo(neighbor).pi for neighbor in self.G.neighbors(node)])
        # setattr(self.indInfo(node), 'xi', self.lambd*abs(individual.pi - mean_polarity_neighbours))    
        # Updates the information distortion probabilities based on entropic effects and polarzation bias
        setattr(individual, 'DistortionProbability', get_transition_probabilities(individual, tendency))   
        
    def initialize_nodes(self):
        """
        Attribute to each node in the network a new instance of 'Individual'.
        """
        nx.set_node_attributes(self.G, 
                               {node : Individual(self.kappa) \
                                                    for node in self.G}, 
                               name = 'Object')
            
    def group_individuals(self):
        """
        Individuals are randomly grouped accordingly to their polarization tendencies.
        """        
        indices = sample(list(range(self.N)), k = self.N)
        group_a = indices[:int(self.alpha*self.N)]
        group_b = indices[int(self.alpha*self.N): int(self.alpha*self.N) + int(self.omega*self.N)]
        group_c = indices[int(self.alpha*self.N) + int(self.omega*self.N):]
        
        attribute_dict = {}
        attribute_dict.update({list(self.G.nodes())[i]:'Up' for i in group_a})
        attribute_dict.update({list(self.G.nodes())[i]:'Down' for i in group_b})
        attribute_dict.update({list(self.G.nodes())[i]:'-' for i in group_c})
        
        nx.set_node_attributes(self.G, attribute_dict, name = 'Tendency')
        
    def compute_sigma_attribute(self):
        """
        Uses the edge weights to measure popularity of each node.
        """        
        for node in self.G:
            global_proximity = sum([self.G.edges[(node, neighbor)]['Distance'] for neighbor in self.G.neighbors(node)])
            setattr(self.indInfo(node), 'sigma', global_proximity)
            
    def compute_edge_weights(self):
        """
        Used the Jensen-Shannon Divergence to attribute edge weights, measuring ideological proximity.
        """        
        nx.set_edge_attributes(self.G, 
                               {(u, v):(1 - JSD(self.indInfo(u).P, self.indInfo(v).P)) \
                                                                    for u, v in self.G.edges}, 
                               'Distance')
            
    def get_acceptance_probability(self, u: int, v:int) -> float:
        """
        Gets the probability that individual "u" will accept an information transmited by "v" based on ideological proximity and perception of relative popularity.

        Args:
            u (int): The vertex index of an individual "u".
            v (int): The vertex index of an indiviudal "v".

        Returns:
            float: The \eta_{u to v} probability.
        """        
        return acceptance_probability(self.G, u, v, self.gamma)
        
    def indInfo(self, node: int) -> Individual:
        """
        Gets the "Individual" instance corresponding to a given node index.

        Args:
            node (int): A vertex index.

        Returns:
            Individual: An instance of "Individual".
        """        
        return getInfo(self.G, node)
    
    def indTendency(self, node: int) -> str:
        """
        Gets the polarization tendency of a given node index.

        Args:
            node (int): A vertex index.

        Returns:
            str: A string defining the polarization tendency of the node (Up, Down or None)
        """        
        return getTendency(self.G, node) 
        
    def compute_graph_entropy(self):
        """
        Computes the mean entropy of the network's individuals.
        """        
        self._H = sum([self.indInfo(node).H for node in self.G])/self.N
        
    def compute_graph_polarity(self):
        """
        Computes the mean polarity of the network's individuals.
        """        
        self._pi = sum([self.indInfo(node).pi for node in self.G])/self.N
        
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
                  gamma: float) -> Dict:
    """
    Evaluate a new model over T iterations.

    Args:
        T (int): Number of iterations to simulate the model.
        kappa (float): Conservation factor.
        lambd (float): Polarization factor.
        alpha (float): Proportion of individuals polarizing upwards.
        omega (float): Proportion of individuals polarizing downwards.
        gamma (float): Confidence coefficient.
        seed (int, optional): Seed for random number generator.. Defaults to 42.

    Returns:
        Dict: A dictionary of statistics extracted from the model after each iteration.
    """    
    print("Initializing model with parameters")
    print("N = {} - pa = {} - mu = {} - m = {} - kappa = {} - lambda = {} - alpha = {} - omega = {} - gamma = {}".format(N, pa, memory_size, code_length, kappa, lambd, alpha, omega, gamma))
    
    model = Model(N, pa, memory_size, code_length, kappa, lambd, alpha, omega, gamma, seed)
    
    statistics = {}
    statistics['H - seed = {}'.format(model.seed)]  = []
    # statistics['pi - seed = {}'.format(model.seed)] = []
    update_statistics(model, statistics)

    elapsedTime = 0

    for i in range(T):
        print(f"Starting iteration {i}")
        execution_time = simulate(model)
        elapsedTime += execution_time
        print(f"Iteration {i} ended - Execution Time = {np.round(execution_time, 5)}")
        update_statistics(model, statistics)
        
    print(f"Simulation ended. Execution time = {np.round(elapsedTime, 2)} min")        
    return elapsedTime, statistics

def simulate(M: Model) -> float:
    """
    Execute one iteration of the information propagation model, updating the model's parameters at the end. Return the execution time (minutes).

    Args:
        M (Model): A model instance.  

    Returns:
        float: Execution time.
    """    
    start = time()
    for u, v in M.G.edges():
        u_ind = M.indInfo(u)
        v_ind = M.indInfo(v)
        u_ind.receive_information(evaluate_information(distort(v_ind.X, v_ind.DistortionProbability), M.get_acceptance_probability(u, v)))
        v_ind.receive_information(evaluate_information(distort(u_ind.X, u_ind.DistortionProbability), M.get_acceptance_probability(v, u)))     
    end = time()
    
    M.update_model()
    return (end - start)/60

def update_statistics(M: Model, statistics: Dict):
    """Updates statistics extracted from the model.

    Args:
        M (Model): A model instance.
        statistics (Dict): A dictionary with the statistics arrays to be updated.
    """ 
    statistics['H - seed = {}'.format(M.seed)].append(M.H)
    # statistics['pi - seed = {}'.format(M.seed)].append(M.pi)