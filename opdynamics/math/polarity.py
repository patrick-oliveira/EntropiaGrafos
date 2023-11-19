import numpy as np

from opdynamics.utils.types import Binary


def polarity_weights(code_length: int, seed: int) -> np.ndarray:
    """
    Return a normalized list of 'm' random values.
    These values are used as weights to calculate the polarity of a
    binary code.

    Args:
        code_length (int): The length of the binary code.

    Returns:
        np.ndarray: A numpy array of 'm' normalized random values.
    """
    np.random.seed(seed)

    v = np.ones(code_length)
    return v/sum(v)


def polarity(x: Binary, code_length: int, seed: int) -> float:
    """
    Return the weighted average of bits using "beta" as weight vector.

    Args:
        x (Binary): A binary code (array of bits)
        beta (Binary): A weight vector (list of bits)

    Returns:
        float: The information's polarity (a weighted average)
    """
    return np.dot(x, polarity_weights(code_length, seed))


def xi(lambd: float) -> float:
    """
    Distortion probability related to the polarization tendency of the
    individual.

    Args:
        lambd (float): The polarization factor defined for the model.

    Returns:
        float: The distortion probability between (0, 0.5).
    """
    return 1 / (np.exp(lambd) + 1)
