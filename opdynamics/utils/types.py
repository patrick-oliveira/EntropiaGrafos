import matplotlib.axes as ax
import matplotlib.figure as fig
import networkx as nx  # type: ignore
import numpy as np

from numpy.typing import NDArray
from pydantic import BaseModel
from typing import (
    Any,
    Dict,
    List,
    NewType,
    TypedDict,
    Optional,
)


class KernelDensityEstimator:
    def __getattr__(self, item: str) -> Any:
        pass


class Memory(TypedDict):
    codes: NDArray
    polarities: NDArray
    distribution: KernelDensityEstimator


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
