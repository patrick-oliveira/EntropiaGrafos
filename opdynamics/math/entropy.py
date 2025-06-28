import numpy as np

from functools import partial
from opdynamics.utils.types import CodeDistribution


def shannon_entropy(P: np.ndarray) -> float:
    """
    Calculate the Shannon entropy of a probability distribution.

    Parameters:
    P (np.ndarray): The probability distribution.

    Returns:
    np.ndarray: The Shannon entropy of the probability distribution.
    """
    P = P[P > 0]
    return - (P * np.log2(P)).sum()


def memory_entropy(distribution: CodeDistribution) -> float:
    """
    Calculate the memory entropy of a given distribution.

    Parameters:
        distribution (CodeDistribution): A dictionary representing the
        probability distribution.

    Returns:
        float: The memory entropy value.

    """
    dist_dict = distribution.distribution

    P = np.asarray(list(dist_dict.values()))
    return shannon_entropy(P)


def JSD(Pu: CodeDistribution, Pv: CodeDistribution) -> float:
    """
    Calculates the Jensen-Shannon Divergence (JSD) between two probability
    distributions.

    Parameters:
        Pu (CodeDistribution): The first probability distribution.
        Pv (CodeDistribution): The second probability distribution.

    Returns:
        float: The JSD value.

    """

    M = CodeDistribution(
        distribution = {
            code: (Pu.distribution[code] + Pv.distribution[code]) / 2
            for code in Pu.distribution.keys()
        }
    )

    return memory_entropy(M) - (memory_entropy(Pu) + memory_entropy(Pv)) / 2


def S(Pu: CodeDistribution, Pv: CodeDistribution) -> float:
    """
    Calculates the similarity between two code distributions using the
    Jenson-Shannon Divergence (JSD).

    Parameters:
    Pu (CodeDistribution): The first code distribution.
    Pv (CodeDistribution): The second code distribution.

    Returns:
    float: The similarity between the two code distributions.
    """
    return 1 - JSD(Pu, Pv)


def D(P: CodeDistribution, Q: CodeDistribution) -> float:
    """
    Calculates the Kullback-Leibler divergence between two code distributions.

    Parameters:
        P (CodeDistribution): The first code distribution.
        Q (CodeDistribution): The second code distribution.

    Returns:
        float: The Kullback-Leibler divergence between P and Q.
    """
    return sum(map(partial(_D, P, Q), P.keys()))


def _D(P: CodeDistribution, Q: CodeDistribution, x: str):
    """
    Calculate the Kullback-Leibler divergence between two code distributions P
    and Q for a given binary value x.

    Parameters:
    P (CodeDistribution): The first code distribution.
    Q (CodeDistribution): The second code distribution.
    x (str): The binary value.

    Returns:
    float: The Kullback-Leibler divergence between P and Q for the given
    binary value x.
    """
    if P.distribution[x] == 0:
        return 0
    else:
        if Q.distribution[x] == 0:
            return np.inf
        else:
            a = P.distribution[x]
            b = np.log2(P.distribution[x] / Q.distribution[x])
            return a * b