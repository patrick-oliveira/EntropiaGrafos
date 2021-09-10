import networkx as nx
from typing import NewType, List, Dict

Graph = NewType('Graph', nx.Graph)
Binary = NewType('Binary', str)
LBinary = NewType('Binary', List[int])
Memory = NewType('Memory', List[LBinary])
CodeDistribution = NewType('CodeDistribution', Dict[Binary, float])
Weights = NewType('Weights', List[float])
TransitionProbabilities = NewType('TransitionProbabilities', Dict[str, float])