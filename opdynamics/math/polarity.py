import numpy as np

from opdynamics import SEED
from opdynamics.utils.types import Binary

def polarity_weights(code_length: int) -> np.ndarray:
    """
    Return a normalized list of 'm' random values.
    These values are used as weights to calculate the polarity of a binary code.

    Args:
        code_length (int): The length of the binary code.

    Returns:
        np.ndarray: A numpy array of 'm' normalized random values.
    """    
    np.random.seed(SEED)
    
    v = np.ones(code_length)
    return v/sum(v)

def polarity(x: Binary, beta: np.ndarray) -> float:
    """
    Return the weighted average of bits using "beta" as weight vector.

    Args:
        x (Binary): A binary code (list of bits)
        beta (Binary): A weight vector (list of bits)

    Returns:
        float: The information's polarity (a weighted average)
    """
    return np.dot(beta, x)