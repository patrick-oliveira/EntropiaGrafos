import numpy as np
from functools import partial
from opdynamics.utils.types import Binary, CodeDistribution


def shannon_entropy(P: float) -> float:
    """
    Calculates the Shannon entropy of a probability distribution.

    Args:
        P (float): The probability of an event.

    Returns:
        float: The Shannon entropy of the probability distribution.
    """
    return P*np.log2(P)


def memory_entropy(distribution: CodeDistribution) -> float:
    """
    Calculates the entropy of a given distribution.

    Parameters:
    distribution (CodeDistribution): A dictionary-like object containing the frequency of each code.

    Returns:
    float: The memory entropy of the distribution.
    """
    P = np.asarray(list(distribution.values()))
    P = P[P > 0]
    return - shannon_entropy(P).sum()
            
def JSD(Pu: CodeDistribution, Pv: CodeDistribution) -> float:
    """
    Calculates the Jensen-Shannon divergence between two probability distributions.

    Parameters:
    Pu (CodeDistribution): The first probability distribution.
    Pv (CodeDistribution): The second probability distribution.

    Returns:
    float: The Jensen-Shannon divergence between the two probability distributions.
    """
    M = {code:(Pu[code] + Pv[code])/2 for code in Pu}
    
    return memory_entropy(M) - (memory_entropy(Pu) + memory_entropy(Pv))/2

def S(Pu: CodeDistribution, Pv: CodeDistribution) -> float:
    """
    Calculates the similarity between two probability distributions using the Jensen-Shannon divergence.

    Args:
        Pu (CodeDistribution): The first probability distribution.
        Pv (CodeDistribution): The second probability distribution.

    Returns:
        float: The similarity between the two probability distributions.
    """
    return 1 - JSD(Pu, Pv)


def D(P: CodeDistribution, Q: CodeDistribution) -> float:
    """
    Calculates the Kullback-Leibler divergence between two probability distributions P and Q.

    Args:
        P (Dict[str, float]): The first probability distribution.
        Q (Dict[str, float]): The second probability distribution.

    Returns:
        float: The Kullback-Leibler divergence between P and Q.
    """
    return sum(map(partial(_D, P, Q), P.keys()))

def _D(P: CodeDistribution, Q: CodeDistribution, x: Binary):
    """
    Auxiliary function.
    
    Calculates the Kullback-Leibler divergence between two probability distributions P and Q for a given binary value x common to both distributions.

    Args:
        P (CodeDistribution): The first probability distribution.
        Q (CodeDistribution): The second probability distribution.
        x (Binary): The binary value for which to calculate the divergence.

    Returns:
        The Kullback-Leibler divergence between P and Q for the given binary value x.
    """
    if P[x] == 0:
        return 0
    else:
        if Q[x] == 0:
            return np.inf
        else:
            return P[x]*np.log2(P[x]/Q[x])