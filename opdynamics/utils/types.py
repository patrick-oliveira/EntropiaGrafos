import matplotlib.axes as ax
import matplotlib.figure as fig
import networkx as nx
import numpy as np

from sklearn.neighbors import KernelDensity
from pydantic import BaseModel
from pydantic_numpy import NpNDArray
from typing import (
    Dict,
    List,
    NewType,
    TypedDict,
    Optional
)


KernelProbabilityDensity = NewType('KernelProbabilityDensity', KernelDensity)


class Memory(TypedDict):
    codes: NpNDArray
    polarities: NpNDArray
    distribution: KernelDensity


CodeDistribution = NewType('CodeDistribution', Dict[str, float])


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

l_multiply = lambda x, y: x * y # noqa
l_sum      = lambda x, y: x + y # noqa


class SimulationResult(TypedDict):
    entropy: np.array
    proximity: np.array
    polarity: np.array
    distribution: np.array
    acceptances: dict
    transmissions: dict
