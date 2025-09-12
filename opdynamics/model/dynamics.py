import numpy as np

from opdynamics import seed
from opdynamics.components import Individual
from opdynamics.math.entropy import JSD
from opdynamics.utils.types import Graph, TransitionProbabilities
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


def get_transition_probabilities(
    ind: Individual,
    tendency: str = None
) -> TransitionProbabilities:
    """
    Return a dictionary with probabilities of bit distortion, i.e. the
    probability of 0 -> 1 and 1 -> 0, considering the individual's tendency.

    Args:
        ind (Individual): An individual object.
        tendency (str): The identification of the individual's tendency
        (polarizing upwards or downwards)

    Returns:
        TransitionProbabilities: The dictionary of probabilities for the
        transitions 0 -> 1 and 1 -> 0.
    """
    return {0: ind.delta + ind.xi, 1: ind.delta} if tendency == 1 else \
           {0: ind.delta, 1: ind.delta + ind.xi} if tendency == -1 else \
           {0: ind.delta, 1: ind.delta}


def distort(
    code: np.ndarray,
    transition_probability: TransitionProbabilities
) -> np.ndarray:
    """
    Distorts the given code using the provided transition probabilities.

    Args:
        code (np.ndarray): The code to be distorted.
        transition_probability (TransitionProbabilities): The transition
        probabilities.

    Returns:
        np.ndarray: The distorted code.
    """
    for k in range(len(code)):
        code[k] = mutate(code[k], transition_probability)

    return code


def mutate(
    bit: int,
    transition_probability: TransitionProbabilities
) -> int:
    """
    Mutates a bit based on a given transition probability.

    Args:
        bit (int): The bit to be mutated.
        transition_probability (TransitionProbabilities): The transition
        probabilities for each bit.

    Returns:
        int: The mutated bit.
    """
    x = np.random.uniform()
    p = transition_probability[bit]
    if x <= p:
        return int(not bit)
    else:
        return bit

def proximity(u: Individual, v:Individual) -> float:
    """
    Return the proximity between individuals u and v based on the Jensen-Shannon Divergence.

    Args:
        u (Individual): An individual 'u'
        v (Individual): An individual 'v'

    Returns:
        float: The Jensen-Shannon Divergence JSD(Pu, Pv), where Pu and Pv are the memory's probability distribution of individuals u and v, respectively.
    """
    return 1 - JSD(u.P, v.P)

def acceptance_probability(G: Graph, u: int, v: int, gamma: float) -> float:
    """
    Return the probability that an individual "u" will accept and information given by "v".

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
        set([G.degree[u]**gamma]).union([G.degree[w]**gamma for w in list(G.neighbors(u))])
    )

    sigma_ratio =(G.degree[v]**gamma)/max_sigma
    return 1/( 1/(G[u][v]['Distance'] + e) + 1/(sigma_ratio + e) )


e = 1e-10