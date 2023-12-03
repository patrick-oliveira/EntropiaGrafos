import numpy as np
from opdynamics.utils.types import Binary, CodeDistribution
from opdynamics import CODE_LENGTH

POWERS_OF_TWO = 2**np.arange(CODE_LENGTH)[::-1]


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
        x (string): A binary code represented as a string.

    Returns:
        int: An integer.
    """
    return int('0b'+x, 2)


def complete_zeros(x: str, m: int) -> str:
    """
    Complete the number of bits in "x" with zeroes.
    This procedure may be necessary in order to create a list of binary codes
    with the same length.

    Args:
        x (str): A binary code.
        m (int): Size of the final binary code.

    Returns:
        Binary: A binary code
    """
    return '0'*(m - len(x))+x if (m - len(x)) >= 0 else x


def string_to_binary(x: str) -> Binary:
    """
    Converts a string to a binary array.

    Args:
        x (str): The string to be converted.

    Returns:
        Binary: A binary array representing the input string.
    """
    return np.asarray(list(x)).astype(int)


def binary_to_int(x: Binary) -> str:
    """
    Converts a binary number to its integer representation.

    Args:
        x (Binary): A binary number represented as a numpy array.

    Returns:
        str: The integer representation of the binary number.
    """
    return (POWERS_OF_TWO*x).sum()


def binary_to_string(x: Binary) -> str:
    """
    Converts a binary array to a string representation.

    Args:
        x (numpy.ndarray): A binary array.

    Returns:
        str: A string representation of the binary array.
    """
    return ''.join(list(x.astype(str)))


def random_selection(distribution: CodeDistribution) -> Binary:
    """
    Randomly selects a code from a distribution based on the probability
    of each code.

    Args:
        distribution (CodeDistribution): A dictionary where the keys are codes
        and the values are their probabilities.

    Returns:
        Binary: The selected code in binary format.
    """
    x = np.random.uniform()
    cumulative_probability = 0
    for code in distribution.keys():
        cumulative_probability += distribution[code]
        if cumulative_probability >= x:
            return string_to_binary(code)
