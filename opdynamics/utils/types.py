from typing import Dict, List, NewType

import matplotlib
import networkx as nx
import numpy as np

Graph = NewType('Graph', nx.Graph)
Array = NewType('Array', np.array)
Binary = NewType('Binary', str)
Memory = NewType('Memory', List[Binary])
CodeDistribution = NewType('CodeDistribution', Dict[Binary, float])
Weights = NewType('Weights', List[float])
TransitionProbabilities = NewType('TransitionProbabilities', Dict[str, float])

Figure = NewType('Figure', matplotlib.figure.Figure)
Axis   = NewType('Figure', matplotlib.axes.Axes)
l_multiply = lambda x, y: x * y
l_sum      = lambda x, y: x + y