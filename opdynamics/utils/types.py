import matplotlib.axes as ax
import matplotlib.figure as fig
import networkx as nx
import numpy as np
from typing import (Dict,
                    List,
                    NewType,
                    TypedDict,
                    Optional)

Binary = NewType('Binary', np.array)
Polarity = NewType("Polarity", np.array)
Memory = NewType('Memory', List[Binary])
CodeDistribution = NewType('CodeDistribution', Dict[str, float])
Graph = NewType('Graph', nx.Graph)

Weights = NewType('Weights', List[float])
TransitionProbabilities = NewType('TransitionProbabilities', Dict[str, float])

Figure = NewType('Figure', fig.Figure)
Axis = NewType('Figure', ax.Axes)


class ExperimentParameters(TypedDict):
    simulation_parameters: Dict[str, List[str | int | float]]
    GeneralParameters: Dict[str, str | int | float]


class SimulationParameters(TypedDict):
    graph_type: str
    network_size: int
    memory_size: int
    code_length: int
    kappa: int
    lambd: int
    alpha: int
    omega: int
    gamma: int
    preferential_attachment: int
    polarization_grouping_type: int
    distribution: Optional[str]
    regular_graph_d: Optional[int]
    erdos_graph_p: Optional[float]
    lam: Optional[float]
    base_list: Optional[List[int]]


class GeneralParameters(TypedDict):
    T: int
    num_repetitions: int
    early_stop: bool
    epsilon: float
    results_path: str


class Parameters(TypedDict):
    simulation_parameters: SimulationParameters
    general_parameters: GeneralParameters
