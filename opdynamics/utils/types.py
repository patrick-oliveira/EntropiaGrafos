import matplotlib.axes as ax
import matplotlib.figure as fig
import networkx as nx
import numpy as np

from pydantic import BaseModel
from numpydantic import NDArray
from typing import (
    Dict,
    List,
    NewType,
    TypedDict,
    Optional
)


class Memory(BaseModel):
    codes: NDArray
    polarities: NDArray


class CodeDistribution(BaseModel):
    distribution: Dict[str, float]


class SimulationParameters(BaseModel):
    graph_type: str
    network_size: int
    memory_size: int
    code_length: int
    kappa: float
    lambd: float
    alpha: float
    omega: float
    gamma: float
    preferential_attachment: int
    polarization_type: int
    distribution: Optional[str]


class GeneralParameters(BaseModel):
    T: int
    num_repetitions: int
    early_stop: bool
    epsilon: float
    results_path: str


class Parameters(BaseModel):
    simulation_parameters: dict
    general_parameters: dict


Binary = str

Graph = NewType('Graph', nx.Graph)
Array = NewType('Array', np.array)
Weights = NewType('Weights', List[float])
TransitionProbabilities = NewType('TransitionProbabilities', Dict[str, float])

Figure = NewType('Figure', fig.Figure)
Axis   = NewType('Figure', ax.Axes)

l_multiply = lambda x, y: x * y
l_sum      = lambda x, y: x + y


class SimulationResult(TypedDict):
    entropy: np.array
    proximity: np.array
    polarity: np.array
    distribution: np.array
    acceptances: dict
    transmissions: dict