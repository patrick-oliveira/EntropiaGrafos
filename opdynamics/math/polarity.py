import numpy as np

from opdynamics import code_length
from opdynamics.utils.types import Weights


def polarity_weights(*args, **kwargs) -> Weights:
    """
    Return a normalized list of 'm' random values.
    These values are used as weights to calculate the polarity of a binary
    code.

    Returns:
        Weights: A numpy array of 'm' normalized random values.
    """
    v = np.ones(code_length)
    return v / sum(v)


def polarity(x: np.ndarray, *args, **kwargs) -> float:
    return [0.5 for _ in x.shape[0]]
