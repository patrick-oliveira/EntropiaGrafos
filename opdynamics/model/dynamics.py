import numpy as np

from typing import Optional
from opdynamics import SEED
# from opdynamics.components import Individual
from opdynamics.math.entropy import JSD
from opdynamics.math.probability import max_sigma, sigma, acc_prob
from opdynamics.utils.types import Binary, Graph, TransitionProbabilities

np.random.seed(SEED)


def evaluate_information(
    code: Binary,
    acceptance_probability: float,
    *args,
    **kwargs
) -> Optional[Binary]:
    """
    Function which decides wether or not the incoming information will be
    accepted. If refused, return "None".

    Args:
        code (Binary): A incoming binary code.
        acceptance_probability (float): Probability of accepting "code"

    Returns:
        Optional[Binary]: The incoming code if accepted, otherwise "None".
    """
    return code if (np.random.uniform() <= acceptance_probability) else None


def get_transition_probabilities(
    delta: float,
    xi: float,
    tendency: str = None
) -> TransitionProbabilities:
    """
    Return a dictionary with probabilities of bit distortion, i.e. the
    probability of 0 -> 1 and 1 -> 0, considering the individual's tendency.

    Args:
        ind (Individual): An Individual object.
        tendency (str): The identification of the individual's tendency
        (polarizing upwards or downwards)

    Returns:
        TransitionProbabilities: The dictionary of probabilities for the
        transitions 0 -> 1 and 1 -> 0.
    """
    return {0: delta + xi, 1: delta} if tendency == 1 else \
           {0: delta, 1: delta + xi} if tendency == -1 else \
           {0: delta, 1: delta}


def distort(
    code: Binary,
    transition_probability: TransitionProbabilities
) -> Binary:
    """
    Return 'code' after bitwise distortion according to the probabilities
    given by "transition_probability".

    Args:
        code (Binary): A binary code.
        transition_probability (TransitionProbabilities): The probabilitions
        for the transitions 0 -> 1 and 1 -> 0.

    Returns:
        Binary: A possibily bitwise distorted code.
    """
    for k in range(len(code)):
        code[k] = mutate(code[k], transition_probability)

    return code


def mutate(
    bit: int,
    transition_probability: TransitionProbabilities
) -> Binary:
    """
    Mutates a given bit based on a transition probability.

    Args:
        bit (int): The bit to mutate.
        transition_probability (TransitionProbabilities): The transition
        probabilities for each bit.

    Returns:
        Binary: The mutated bit.
    """
    x = np.random.uniform()
    p = transition_probability[bit]
    if x <= p:
        return int(not bit)
    else:
        return bit


def proximity(
    u,
    v,
) -> float:
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


def acceptance_probability(
    G: Graph,
    u: int,
    v: int,
    gamma: float
) -> float:
    """
    Return the probability that an individual "u" will accept and information
    given by "v".

    Args:
        G (Graph): The Graph model.
        u (int): The vertex id of an individual 'u'.
        v (int): The vertex id of an individual 'v'.
        gamma (float): The 'confidence factor' parameter.

    Returns:
        float: The acceptance probability for u of informations given by v.
    """

    # measures the popularity of the receiver and its neighbors and sender
    u_ms = max_sigma(
        G.degree[u],
        [G.degree[w] for w in list(G.neighbors(u))],
        gamma
    )
    v_s = sigma(G.degree[v], gamma)

    # measures the popularity of the sender in relation to the receiver and
    # its neighbors
    sigma_ratio = v_s/u_ms

    return acc_prob(sigma_ratio, G[u][v]['Distance'])
