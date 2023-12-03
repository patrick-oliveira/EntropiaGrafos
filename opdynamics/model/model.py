import networkx as nx

from typing import Dict
from opdynamics.utils.types import Graph
from opdynamics.components import Individual
from opdynamics.math.entropy import S
from opdynamics.math.polarity import xi
from opdynamics.model.utils import (order_indexes,
                                    group_indexes)
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
        seed: int,
        preferential_attachment: int,
        polarization_grouping_type: int = 0,
        regular_graph_d: int = None,
        erdos_graph_p: float = None,
        initialize: bool = True,
        distribution: str = "binomial",
        verbose: bool = False,
        *args,
        **kwargs
    ):
        # General Parameters
        self.seed = seed

        # Network parameters
        self.N = network_size
        self.pa = preferential_attachment
        self.regular_graph_d = regular_graph_d
        self.erdos_graph_p = erdos_graph_p
        self.graph_type = graph_type

        # Individual parameters
        self.mu = memory_size
        self.m = code_length
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
        if initialize:
            self.initialize_model_info()

    @property
    def H(self) -> float:
        """
        Mean Entropy

        Returns:
            [float]: Mean entropy of all the individuals
        """
        return self._H

    @property
    def S(self) -> float:
        """
        Mean weight of network edges.

        Returns:
            float: Mean weight of all edges in the network.
        """
        return self._S

    @property
    def pi(self) -> float:
        """
        Mean polarity.

        Returns:
            float: Mean polarity value of all individuals.
        """
        return self._pi

    @property
    def G(self) -> Graph:
        """
        The model's network.

        Returns:
            Graph: The model's network as a networkx Graph object.
        """
        return self._G

    @property
    def ind_vertex_objects(self) -> Dict[int, Individual]:
        """
        The model's individuals.

        Returns:
            Dict[int, Individual]: A dictionary containing the model's
            individuals as values and their corresponding vertex index as keys.
        """
        return self._ind_vertex_objects

    @property
    def vertex_tendencies(self) -> Dict[int, int]:
        """
        The individual's tendencies.

        Returns:
            Dict[int, int]: A dictionary containing the model's individuals
            tedencies as values and their corresponding vertex index as keys.
        """
        return self._vertex_tendencies

    def create_graph(self):
        """
        Creates a network following the specified topology.
        """
        if self.graph_type == 'barabasi':
            self._G = nx.barabasi_albert_graph(
                self.N,
                self.pa,
                self.seed
            )
        elif self.graph_type == 'complete':
            self._G = nx.complete_graph(self.N)
        elif self.graph_type == 'regular':
            self._G = nx.random_regular_graph(
                n=self.N,
                d=self.regular_graph_d,
                seed=self.seed
            )
        elif self.graph_type == 'erdos':
            self._G = nx.erdos_renyi_graph(
                n=self.N,
                p=self.erdos_graph_p,
                seed=self.seed
            )

    def initialize_model_info(self):
        """
        Initialize the model by attributing to each node an 'Individual'
        object, grouping them based on their polarization tendencies, and
        computing the necessary parameters and measures.
        """
        self.initialize_nodes()
        self.group_individuals()
        self.compute_model_measures()

    def initialize_nodes(self):
        """
        Attribute to each node in the network a new instance of 'Individual'.
        """
        nx.set_node_attributes(
            self.G,
            {
                node: Individual(
                    kappa=self.kappa,
                    memory_size=self.mu,
                    distribution=self.distribution,
                    code_length=self.m,
                    seed=None,
                    *self.args,
                    **self.kwargs
                ) for node in self.G},
            name='Object'
        )
        self._ind_vertex_objects = nx.get_node_attributes(self.G, 'Object')

    def group_individuals(self):
        """
        Attributes to each node in the network a polarization tendency
        (positive, negative or neutral).
        The number of individuals in each group is defined by the parameters
        alpha and omega.
        The individuals are grouped based on their degree, which is defined by
        the network topology, following
        the function "order_indexes" in the "utils.py" file:
            - If polarization_grouping_type = 0, the individuals are randomly
            grouped.
            - If polarization_grouping_type = 1, the most alpha*N most
            connected individuals are polarized positively, the next omega*N
            most connected individuals are polarized negatively, and the
            remaining individuals are neutral.
            - If polarization_grouping_type = 2, the alpha*N least connected
            individuals are polarized positively, the next omega*N least
            connected individuals are polarized negatively, and the remaining
            individuals are neutral.
        """
        indexes = order_indexes(
            self.N,
            self.polarization_grouping_type,
            self.G.degree
        )

        groups = group_indexes(indexes, self.alpha, self.omega, self.N)

        nodes = list(self.G.nodes())
        attribute_dict = {}
        for label, group in groups.items():
            attribute_dict.update({nodes[i]: label for i in group})
        nx.set_node_attributes(self.G, attribute_dict, name='Tendency')

        self._vertex_tendencies = nx.get_node_attributes(self.G, 'Tendency')

    def compute_model_measures(self):
        """
        Computes various measures of the model, including edge weights, sigma
        attribute, graph entropy, mean edge weight, and graph polarity.
        """
        self.compute_edge_weights()
        self.compute_ind_attributes()
        self.compute_graph_entropy()
        self.compute_mean_edge_weight()
        self.compute_graph_polarity()

    def compute_edge_weights(self):
        """
        Used the Jensen-Shannon Divergence to attribute edge weights,
        measuring ideological proximity.
        """
        nx.set_edge_attributes(
            self.G,
            {
                (u, v): S(self.indInfo(u).P, self.indInfo(v).P)
                for u, v in self.G.edges
            },
            'Distance'
        )

    def compute_ind_attributes(self):
        """
        Computes individual attributes for each node in the graph.

        For each node in the graph, this method computes the individual's
        sigma and xi attributes, as well as their distortion probability based
        on their individual tendency.
        """
        for node in self.G:
            individual = self.indInfo(node)
            tendency = self.indTendency(node)
            neighbors = self.G.neighbors(node)

            global_prox = sum(
                [self.G.edges[(node, n)]['Distance'] for n in neighbors]
            )

            setattr(individual, 'sigma', global_prox)
            setattr(individual, 'xi', xi(self.lambd))
            setattr(
                individual,
                'DistortionProbability',
                get_transition_probabilities(
                    individual.delta,
                    individual.xi,
                    tendency
                )
            )

    def compute_graph_entropy(self) -> None:
        """
        Computes the mean entropy of the network's individuals.
        """
        self._H = sum([self.indInfo(node).H for node in self.G]) / self.N

    def compute_mean_edge_weight(self) -> None:
        '''
        Computes the mean distance of the networks edges.
        '''
        self._S = sum(
            [self.G[u][v]['Distance'] for u, v in self.G.edges]
        ) / self.E

    def compute_graph_polarity(self) -> None:
        """
        Computes the mean polarity of the network's individuals.
        """
        self._pi = sum([self.indInfo(node).pi for node in self.G]) / self.N

    def update_model(self):
        """
        Updates the network information (individual's information and all
        measures based on entropy) after each iteration.
        """
        for node in self.G:
            self.update_node_info(node, update_memory=True)

        self.compute_model_measures()

    def update_node_info(self, node: int, update_memory: bool = False):
        """
        A function used to update the state of each node in the network.
        After each iteration, this function is called with
        update_memory = True to possibly include new information in each
        individual's memory. After that, polarization and distortion
        probabilities are updated.

        Args:
            node (int): A vertex id.
            update_memory (bool, optional): A boolean which specifies if the
            individual's memory must be updated or not. Defaults to False.
        """
        individual = self.indInfo(node)

        if update_memory:
            individual.update_memory()

    def get_acceptance_probability(self, u: int, v: int) -> float:
        """
        Gets the probability that individual "u" will accept an information
        transmited by "v" based on ideological proximity and perception of
        relative popularity.

        Args:
            u (int): The vertex index of an individual "u".
            v (int): The vertex index of an indiviudal "v".

        Returns:
            float: The \\eta_{u to v} probability.
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
            str: A string defining the polarization tendency of the node (Up,
            Down or None)
        """
        return self.vertex_tendencies[node]
