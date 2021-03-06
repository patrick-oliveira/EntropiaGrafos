import networkx as nx
from typing import NewType, List, Dict

Graph = NewType('Graph', nx.Graph)
Binary = NewType('Binary', str)
Memory = NewType('Memory', List[Binary])
CodeDistribution = NewType('CodeDistribution', Dict[Binary, float])
Weights = NewType('Weights', List[float])
TransitionProbabilities = NewType('TransitionProbabilities', Dict[str, float])