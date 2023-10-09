from random import sample

import networkx as nx
import numpy as np

from opdynamics import seed
from opdynamics.components import Individual
from opdynamics.math.entropy import S
from opdynamics.model.dynamics import (acceptance_probability,
                                       get_transition_probabilities)


class Model:
    def __init__(
        self, 
        graph_type: str, 
        network_size: int,
        memory_size: int, 
        code_length: int,
        kappa: float,
        lambd: float,
        alpha: float, 
        omega: float,
        gamma: float,
        preferential_attachment: int, 
        polarization_grouping_type: int = 0,
        d: int = None, 
        p: float = None,
        initialize: bool = True,
        distribution: str = "binomial",
        *args,
        **kwargs
    ):
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
        self.N  = network_size
        self.pa = preferential_attachment
        self.d  = d
        self.p  = p
        self.graph_type = graph_type
        
        # Individual parameters
        self.mu    = memory_size
        self.m     = code_length
        self.kappa = kappa
        self.alpha = alpha
        self.omega = omega
        self.lambd = lambd
        self.gamma = gamma
        self.distribution = distribution
        
        self.polarization_grouping_type = polarization_grouping_type
        
        self.args = args
        self.kwargs = kwargs
        
        self.create_graph()
        self.E = self.G.number_of_edges()
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
    def J(self) -> float:
        return self._J
    
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
        return self._G
    
    @property
    def ind_vertex_objects(self):
        return self._ind_vertex_objects
    
    @property
    def vertex_tendencies(self):
        return self._vertex_tendencies
        
    def create_graph(self):
        """
        Edite esta função para permitir a criação de redes com topologias diferentes. A função deve receber um nome identificando o tipo de rede, e/ou uma função de criação da rede.
        """ 
        if self.graph_type == 'barabasi':
            self._G = nx.barabasi_albert_graph(self.N, self.pa, seed)
        elif self.graph_type == 'complete':
            self._G = nx.complete_graph(self.N)
        elif self.graph_type == 'regular':
            self._G = nx.random_regular_graph(n = self.N, d = self.d, seed = seed)
        elif self.graph_type == 'erdos':
            self._G = nx.erdos_renyi_graph(n = self.N, p = self.p, seed = seed)
        
    def compute_model_measures(self):
        self.compute_edge_weights()
        self.compute_sigma_attribute()
        self.compute_graph_entropy()
        self.compute_mean_edge_weight()
        self.compute_graph_polarity()
        
    def initialize_model_info(self):
        """
        Initialize the model by creating the individual's information and computing the necessary parameters and measures.
        """        
        self.initialize_nodes()
        self.group_individuals()
        self.compute_model_measures()
        
    def update_model(self):
        """
        Updates the network information (individual's information and all measures based on entropy) after each iteration.
        """        
        
        for node in self.G:
            self.update_node_info(node, update_memory = True)
        
        self.compute_model_measures()
        
    def update_node_info(self, node: int, update_memory: bool = False):
        """
        A function used to update the state of each node in the network. 
        After each iteration, this function is called with update_memory = True to possibly include new information in each individual's memory. After that, polarization and distortion probabilities are updated.

        Args:
            node (int): A vertex id.
            update_memory (bool, optional): A boolean which specifies if the individual's memory must be updated or not. Defaults to False.
        """        
        individual = self.indInfo(node)
        
        if update_memory:
            individual.update_memory()
        
    def initialize_nodes(self):
        """
        Attribute to each node in the network a new instance of 'Individual'.
        """
        nx.set_node_attributes(
            self.G, 
            {node : Individual(
                self.kappa, 
                self.mu, 
                self.distribution,
                *self.args,
                **self.kwargs
            ) for node in self.G}, 
            name = 'Object'
        )
        self._ind_vertex_objects = nx.get_node_attributes(self.G, 'Object')
            
    def group_individuals(self):
        """
        Individuals are randomly grouped accordingly to their polarization tendencies.
        """    
        if self.polarization_grouping_type == 0: # totally random    
            indices = sample(list(range(self.N)), k = self.N)
        elif self.polarization_grouping_type == 1: # most connected individuals are polarized
            indices = list(np.argsort([self.G.degree[x] for x in range(self.N)]))
            indices.reverse()
        elif self.polarization_grouping_type == 2: # less connected individuals are polarized
            indices = list(np.argsort([self.G.degree[x] for x in range(self.N)]))
        
        group_alpha = indices[:int(self.alpha*self.N)]
        group_omega = indices[int(self.alpha*self.N): int(self.alpha*self.N) + int(self.omega*self.N)]
        group_neutral = indices[int(self.alpha*self.N) + int(self.omega*self.N):]
        
        attribute_dict = {}
        attribute_dict.update({list(self.G.nodes())[i]:1 for i in group_alpha})
        attribute_dict.update({list(self.G.nodes())[i]:-1 for i in group_omega})
        attribute_dict.update({list(self.G.nodes())[i]:0 for i in group_neutral})
        
        nx.set_node_attributes(self.G, attribute_dict, name = 'Tendency')
        self._vertex_tendencies = nx.get_node_attributes(self.G, 'Tendency')
        
    def compute_sigma_attribute(self):
        """
        Uses the edge weights to measure popularity of each node.
        """        
        for node in self.G:
            individual = self.indInfo(node)
            tendency   = self.indTendency(node)
            global_proximity = sum([self.G.edges[(node, neighbor)]['Distance'] for neighbor in self.G.neighbors(node)])
            setattr(individual, 'sigma', global_proximity)
            setattr(individual, 'xi', 1 / (np.exp(self.lambd) + 1))  
            setattr(individual, 'DistortionProbability', get_transition_probabilities(individual, tendency)) 
            
            
    def compute_edge_weights(self):
        """
        Used the Jensen-Shannon Divergence to attribute edge weights, measuring ideological proximity.
        """        
        nx.set_edge_attributes(self.G, 
                               {(u, v):(S(self.indInfo(u).P, self.indInfo(v).P)) \
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
        return self.ind_vertex_objects[node]
    
    def indTendency(self, node: int) -> str:
        """
        Gets the polarization tendency of a given node index.

        Args:
            node (int): A vertex index.

        Returns:
            str: A string defining the polarization tendency of the node (Up, Down or None)
        """        
        return self.vertex_tendencies[node]
        
    def compute_graph_entropy(self) -> None:
        """
        Computes the mean entropy of the network's individuals.
        """        
        self._H = sum([self.indInfo(node).H for node in self.G])/self.N
        
    def compute_mean_edge_weight(self) -> None:
        '''
        Computes the mean distance of the networks edges.

        Returns
        -------
        None.
        '''
        self._J = sum([self.G[u][v]['Distance'] for u, v in self.G.edges])/self.E
        
    def compute_graph_polarity(self) -> None:
        """
        Computes the mean polarity of the network's individuals.
        """        
        self._pi = sum([self.indInfo(node).pi for node in self.G])/self.N