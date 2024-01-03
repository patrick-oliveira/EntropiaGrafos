import numpy as np
from opdynamics.utils.types import CodeDistribution
from opdynamics import code_length


POWERS_OF_TWO = 2**np.arange(code_length)[::-1]


def to_bin(x: int) -> str:
    """
    Converts an integer to its binary representation.

    Args:
        x (int): The integer to be converted.

    Returns:
        str: The binary representation of the integer.
    """
    return bin(x)[2:]


def to_int(x: str) -> int:
    """
    Convert the binary code "x" represented as a string to its correspondent
    integer.

    Args:
        x (str): A binary code represented as a string.

    Returns:
        int: An integer representing the converted binary code.
    """
    return int('0b' + x, 2)


def complete_zeros(x: str, m: int) -> str:
    """
    Complete the number of bits in "x" with zeroes.
    This procedure may be necessary in order to create a list of binary codes
    with the same length.

    Args:
        x (str): A binary code.
        m (int): Size of the final binary code.

    Returns:
        str: A binary code with the same length as specified by "m".
    """
    return '0' * (m - len(x)) + x if (m - len(x)) >= 0 else x


def string_to_binary(x: str) -> np.ndarray:
    """
    Converts a string to a binary array.

    Args:
        x (str): The string to be converted.

    Returns:
        np.ndarray: A binary array representing the input string.
    """
    return np.asarray(list(x)).astype(int)


def binary_to_int(x: np.ndarray) -> str:
    """
    Converts a binary number to its integer representation.

    Args:
        x (Binary): A binary number represented as a numpy array.

    Returns:
        str: The integer representation of the binary number.
    """
    return (POWERS_OF_TWO * x).sum()


def binary_to_string(x: np.ndarray) -> str:
    """
    Converts a binary array to a string representation.

    Args:
        x (numpy.ndarray): A binary array.

    Returns:
        str: A string representation of the binary array.
    """
    return ''.join(list(x.astype(str)))


def random_selection(distribution: CodeDistribution) -> np.ndarray:
    """
    Randomly selects a code from a distribution based on the probability
    of each code.

    Args:
        distribution (CodeDistribution): A dictionary where the keys are codes
            and the values are their probabilities.

    Returns:
        np.ndarray: The selected code in binary format.
    """
    dist_dict = distribution.distribution

    x = np.random.uniform()
    cumulative_probability = 0
    for code in dist_dict.keys():
        cumulative_probability += dist_dict[code]
        if cumulative_probability >= x:
            return string_to_binary(code)
