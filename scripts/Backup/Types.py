import graph_tool as gt
import numpy as np
from typing import NewType, List, Dict, Tuple

Graph = NewType('Graph', gt.Graph)
Array = NewType('Array', np.array)
Binary = NewType('Binary', List[int])      # Isso aqui não é List[int], mas preciso aprender a definir Array[int]
Memory = NewType('Memory', List[Binary])  # Isso aqui não é List[Binary], mas preciso aprender a definir Array[Binary]
CodeDistribution = NewType('CodeDistribution', Dict[Binary, float])
Weights = NewType('Weights', List[float])
TransitionProbabilities = NewType('TransitionProbabilities', Dict[str, float])