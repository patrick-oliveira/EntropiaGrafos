from typing import Dict, List, NewType, TypedDict

import matplotlib.axes as ax
import matplotlib.figure as fig
import networkx as nx
import numpy as np

Binary = NewType('Binary', np.array)
Polarity = NewType("Polarity", np.array)
Memory = NewType('Memory', List[Binary])
CodeDistribution = NewType('CodeDistribution', Dict[str, float])

Graph = NewType('Graph', nx.Graph)
Weights = NewType('Weights', List[float])
TransitionProbabilities = NewType('TransitionProbabilities', Dict[str, float])

Figure = NewType('Figure', fig.Figure)
Axis   = NewType('Figure', ax.Axes)

l_multiply = lambda x, y: x * y
l_sum      = lambda x, y: x + y

class Parameters(TypedDict):
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
    polarization_type: int
    
class SimulationResult(TypedDict):
    entropy: np.array
    proximity: np.array
    polarity: np.array
    distribution: np.array
    acceptances: dict
    transmissions: dict