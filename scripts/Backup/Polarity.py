import numpy as np
from Types import Weights, Binary
from Parameters import code_length, seed

def polarity_weights() -> Weights:
    """
    Return a normalized list of 'm' random values.
    These values are used as weights to calculate the polarity of a binary code.

    Returns:
        Weights: A numpy array of 'm' normalized random values.
    """    
    v = abs(np.random.randn(code_length))
    return v/sum(v)

def polarity(x: Binary) -> float:
    """Return the weighted average of bits using "beta" as weight vector.

    Args:
        x (Binary): A binary code (numpy array of bits) or a list of binary codes

    Returns:
        float: The information's polarity (a weighted average)
    """ 
    return np.dot(beta, x.T)

np.random.seed(seed)
beta = polarity_weights()