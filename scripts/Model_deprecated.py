from graph_tool.generation import price_network
import numpy as np
import time
from random import sample
from copy import deepcopy
from functools import partial

from Scripts.Types import Dict, List, Tuple
from Scripts.Individual import Individual
from Scripts.ModelDynamics import acceptance_probability, get_transition_probabilities, evaluate_information, distort
from Scripts.Entropy import JSD
from Scripts.Parameters import memory_size, code_length, seed, N
from Scripts.Statistics import StatisticHandler, MeanEntropy, MeanProximity, InformationDistribution, MeanDelta

class Model:
    def __init__(self, N: int,                              # Graph Parameters
                       mu: int, m: int,                     # Memory Parameters
                       kappa: float, lambd: float,          # Proccess Parameters
                       alpha: float, omega: float,          # Population Parameters
                       gamma: float,                        # Information Dissemination Parameters
                       initialize: bool = True,
                       prefferential_att: int = 1):
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
        self.N = N
        self.prefferential_att = prefferential_att
        
        # Individual parameters
        self.mu = mu
        self.m = m
        self.kappa = kappa
        self.alpha = alpha
        self.omega = omega
        self.lambd = lambd
        self.gamma = gamma
        
        self.create_graph()
        if initialize: self.initialize_model_information()
        
    @property
    def G(self):
        return self._G
    
    @property
    def E(self):
        return self._E
    
    @property
    def ind_vertex_object(self):
        return self._ind_vertex_object
    
    @property
    def vertex_tendency_property(self):
        return self._vertex_tendency_property
    
    @property
    def sigma_property(self):
        return self._sigma_property
    
    @property
    def edge_weight_property(self):
        return self._edge_weight_property
    
    @property
    def node_degrees(self):
        return self._node_degrees
    
    @property
    def H(self):
        return self._H

    @property
    def J(self):
        return self._J 
        
    def create_graph(self):
        self._G = price_network(N = self.N, gamma = self.prefferential_att, directed = False)
        self._E = self.G.num_edges()
        self._node_degrees = self.G.get_total_degrees(self.G.get_vertices())
        
    def initialize_model_information(self):
        """
        Initialize the model by creating the individual's information and computing the necessary parameters and measures.
        """        
        self.initialize_nodes()
        self.group_individuals()
        
        for node in self.G.iter_vertices():
            self.update_node_information(node)
            
        self.compute_model_measures()
        
    def initialize_nodes(self):
        """
        Attribute to each node in the network a new instance of 'Individual'.
        """
        self._ind_vertex_object= np.asarray([Individual(self.kappa) for n in range(self.N)])
        self._vertex_tendency_property = self.G.new_vertex_property("short")
        self._sigma_property = self.G.new_vertex_property("double")
        self._edge_weight_property = self.G.new_edge_property("double")
        
    def group_individuals(self):
        """
        Individuals are randomly grouped accordingly to their polarization tendencies.
        """        
        indices = np.asarray(sample(list(range(self.N)), k = self.N))
        
        positive_tendency = indices[:int(self.alpha*self.N)]
        negative_tendency = indices[int(self.alpha*self.N):int(self.alpha*self.N) + int(self.omega*self.N)]
        neutral_tendency  = indices[int(self.alpha*self.N) + int(self.omega*self.N):]
        
        vertices = np.zeros(self.N)
        vertices[positive_tendency] = 1
        vertices[negative_tendency] = -1
        vertices[neutral_tendency]  = 0
        
        self.vertex_tendency_property.a = vertices
        
        
    def update_node_information(self, node: int, update_memory: bool = False):
        """
        A function used to update the state of each node in the network. 
        After each iteration, this function is called with update_memory = True to possibly include new information in each individual's memory. After that, polarization and distortion probabilities are updated.

        Args:
            node (int): A vertex id.
            update_memory (bool, optional): A boolean which specifies if the individual's memory must be updated or not. Defaults to False.
        """        
        individual = self.ind_vertex_object[node]
        tendency   = self.vertex_tendency_property[node]
        
        if update_memory:
            individual.update_memory()
            
        # compute polarity probability
        # mean_polarity_neighbours = np.mean([self.indInfo(neighbor).pi for neighbor in self.G.neighbors(node)])
        # setattr(self.indInfo(node), 'xi', self.lambd*abs(individual.pi - mean_polarity_neighbours))    
        # Updates the information distortion probabilities based on entropic effects and polarzation bias
        setattr(individual, 'DistortionProbability', get_transition_probabilities(individual, tendency))
        
    def update_model(self):
        """
        Updates the network information (individual's information and all measures based on entropy) after each iteration.
        """   
        for node in self.G.iter_vertices():
            self.update_node_information(node, update_memory = True)
        self.compute_model_measures()
    
    def compute_model_measures(self):
        self.compute_edge_weights()
        self.compute_sigma_attribute()
        self.compute_graph_entropy()
        self.compute_mean_edge_weight()
        # self.compute_graph_polarity()
        
    def compute_edge_weights(self):
        """
        Used the Jensen-Shannon Divergence to attribute edge weights, measuring ideological proximity.
        """        
        self.edge_weight_property.a = np.apply_along_axis(self._compute_edge_weight, 1, self.G.get_edges())
        
    def _compute_edge_weight(self, edge: np.array) -> float:
        u = self.ind_vertex_object[edge[0]]
        v = self.ind_vertex_object[edge[1]]
        
        distance = 1 - JSD(u.P, v.P)
        
        return distance
        
    def compute_sigma_attribute(self):
        """
        Uses the edge weights to measure popularity of each node.
        """        
        vertices = np.expand_dims(self.G.get_vertices(), 1)
        self.sigma_property.a = np.apply_along_axis(self._compute_sigma_attribute, 1, vertices)
        
    def _compute_sigma_attribute(self, node: int) -> float:
        neighbors_info = self.G.get_all_edges(node, eprops = [self.edge_weight_property])
        return neighbors_info.T[2].sum()
        
    def compute_graph_entropy(self):
        """
        Computes the mean entropy of the network's individuals.
        """        
        self._H = sum([ind.H for ind in self.ind_vertex_object])/self.N
    
    def compute_mean_edge_weight(self):
        '''
        Computes the mean distance of the networks edges.
        '''
        self._J = (self.edge_weight_property.a.sum()/self.E).item()
        
    def compute_graph_polarity(self) -> None:
        return None
    
    def get_acceptance_probability(self, edge: np.array) -> float:
        """
        Gets the probability that individual "u" will accept an information transmited by "v" based on ideological proximity and perception of relative popularity.

        Args:
            u (int): The vertex index of an individual "u".dd
            v (int): The vertex index of an indiviudal "v".

        Returns:
            float: The \eta_{v to u} probability.
        """      
        u = edge[0]
        v = edge[1]
        edge = self.G.edge(u, v)
        distance = np.max([self.edge_weight_property[edge], epsilon])
        return distance
#         sigma_ratio = np.max([self.calc_sigma_ratio(u, v), epsilon])
#         return 2/( 1/distance + 1/sigma_ratio)
        
    def calc_sigma_ratio(self, u: int, v: int) -> float:
        calc_sigma = lambda x: (self.node_degrees[x] + epsilon)**self.gamma
        
        neighbor_max_sigma = calc_sigma(self.G.get_all_neighbors(u)).max()
        u_sigma        = calc_sigma(u)
        max_sigma = np.max([neighbor_max_sigma, u_sigma])
        
        sigma_ratio = calc_sigma(v) / max_sigma
        
        return sigma_ratio
    
    
def initialize_model(N: int, prefferential_att: float,
                     memory_size: int, code_length: int,
                     kappa: float, lambd: float,
                     alpha: float, omega: float,
                     gamma: float) -> Model:
    print("Initializing model with parameters")
    print("N = {} - pa = {} \nmu = {} - m = {} \nkappa = {} - lambda = {} \nalpha = {} - omega = {} \ngamma = {}".format(N, prefferential_att, memory_size, code_length, kappa, lambd, alpha, omega, gamma))
    
    start = time.time()
    initial_model = Model(N, memory_size, code_length, kappa, lambd, alpha, omega, gamma, prefferential_att = prefferential_att)
    model_initialization_time = time.time() - start
    
    print(f"Model initialized. Elapsed time: {np.round(model_initialization_time/60, 2)} minutes")
    
    return initial_model    
    
def evaluateModel(initial_model: Model,
                  T: int, num_repetitions: int = 1) -> Tuple[float, List[Dict], Dict]:
    """
    Evaluate a new model over T iterations.
    """    
    print("Model evaluation started.")
    print(f"Number of repetitions = {num_repetitions}")
    print("\nStarting simulations.\n")
    simulation_time = []
    statistic_handler = StatisticHandler()
    statistic_handler.new_statistic('Entropy', MeanEntropy())
    statistic_handler.new_statistic('Proximity', MeanProximity())
    statistic_handler.new_statistic('Delta', MeanDelta())
    statistic_handler.new_statistic('Distribution', InformationDistribution())
    
    for repetition in range(1, num_repetitions + 1):
        print(f"Repetition {repetition}/{num_repetitions}")
        
        model = deepcopy(initial_model)
        
        start = time.time()
        for i in range(T):
            simulate(model)
            statistic_handler.update_statistics(model)
        repetition_time = time.time() - start
        simulation_time.append(repetition_time)
        
        print(f"\tFinished repetition {repetition}/{num_repetitions}. Elapsed time: {np.round(simulation_time[-1]/60, 2)} minutes")
        
        statistic_handler.end_repetition()
        
    elapsedTime = sum(simulation_time)
    
    mean_statistics = statistic_handler.get_rep_mean()
    rep_statistics  = statistic_handler.get_statistics(rep_stats = True)
    
    return elapsedTime, rep_statistics, mean_statistics

def simulate(M: Model):
    """
    Execute one iteration of the information propagation model, updating the model's parameters at the end. 
    Return the execution time (minutes).

    Args:
        M (Model): A model instance.  

    Returns:
        None.
    """
    edges = M.G.get_edges()
    f = partial(_get_ind_object, M = M)
    ind_objects = np.apply_along_axis(f, 1, edges)
    for k, uv in enumerate(ind_objects):
        u = uv[0]
        v = uv[1]
        u.receive_information(evaluate_information(distort(v.X, v.DistortionProbability), M.get_acceptance_probability(edges[k])))
        v.receive_information(evaluate_information(distort(u.X, u.DistortionProbability), M.get_acceptance_probability(edges[k][::-1])))
    
    M.update_model()
    
def _get_ind_object(edge: np.array, M: Model) -> np.array:
    return M.ind_vertex_object[edge]
        
        
                                    
epsilon = np.finfo(float).eps