import numpy as np

from opdynamics import seed
from opdynamics.components import Individual
from opdynamics.math.entropy import JSD
from opdynamics.utils.types import Graph, Memory
from opdynamics.math.utils import generate_random_samples
from typing import Union

np.random.seed(seed)


def evaluate_information(
    code: np.ndarray,
    acceptance_probability: float
) -> Union[None, np.ndarray]:
    """
    Function which decides wether or not the incoming information will be
    accepted. If refused, return "None"

    Args:
        code (np.ndarray): A incoming binary code.
        acceptance_probability (float): Probability of accepting "code"

    Returns:
        [np.ndarray]: The incoming binary code, if accepted, otherwise "None".
    """
    return code if (np.random.uniform() <= acceptance_probability) else None


def distort(
    code: np.ndarray,
    ind_tendency: int,
    memory: Memory,
    noise_distribution: str = "multivariate_normal",
    *args,
    **kwargs,
) -> np.ndarray:
    polarities = memory["polarities"]
    codes = memory["codes"]

    noise = generate_random_samples(
        n_samples = 1,
        n_dimensions = code.shape[1],
        distribution = noise_distribution,
    )

    if ind_tendency == 1:
        max_polarity = np.argmax(polarities)
        noise = noise + (code - codes[max_polarity])
    elif ind_tendency == -1:
        min_polarity = np.argmin(polarities)
        noise = noise + (code - codes[min_polarity])

    return code + noise


def proximity(u: Individual, v: Individual) -> float:
    """
    Return the proximity between individuals u and v based on the
    Jensen-Shannon Divergence.

    Args:
        u (Individual): An individual 'u'
        v (Individual): An individual 'v'

    Returns:
        float: The Jensen-Shannon Divergence JSD(Pu, Pv), where Pu and Pv are
        the memory's probability distribution of individuals u and v,
        respectively.
    """
    return 1 - JSD(u.P, v.P)


def acceptance_probability(G: Graph, u: int, v: int, gamma: float) -> float:
    """
    Return the probability that an individual "u" will accept and information
    given by "v".

    ==>> Write the latex formula here. <<==

    Args:
        G (Graph): The Graph model.
        u (int): The vertex id of an individual 'u'.
        v (int): The vertex id of an individual 'v'.
        gamma (float): The 'confidence factor' parameter.

    Returns:
        float: The acceptance probability for u of informations given by v.
    """

    max_sigma = max(
        set([G.degree[u]**gamma]).union(
            [G.degree[w]**gamma for w in list(G.neighbors(u))]
        )
    )

    sigma_ratio = (G.degree[v]**gamma) / max_sigma
    return 2 / (1 / (G[u][v]['Distance'] + e) + 1 / (sigma_ratio + e))


e = 1e-10
