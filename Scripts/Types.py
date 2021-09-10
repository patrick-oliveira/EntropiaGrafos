import networkx as nx
import numpy as np
from typing import NewType, List, Dict

Graph = NewType('Graph', nx.Graph)
Array = NewType('Array', np.array)
Binary = NewType('Binary', str)
Memory = NewType('Memory', List[Binary])
CodeDistribution = NewType('CodeDistribution', Dict[Binary, float])
Weights = NewType('Weights', List[float])
TransitionProbabilities = NewType('TransitionProbabilities', Dict[str, float])