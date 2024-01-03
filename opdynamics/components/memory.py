import numpy as np

from copy import deepcopy
from opdynamics import code_length
from opdynamics.math.polarity import polarity
from opdynamics.utils.types import CodeDistribution, Memory
from opdynamics.components.utils import (
    complete_zeros,
    to_bin,
    string_to_binary,
    binary_to_string,
    POWERS_OF_TWO
)


def initialize_memory(
    memory_size: int,
    distribution: str = "binomial",
    *args,
    **kwargs
) -> Memory:
    """
    Create a list of size "mu" of random binary codes of an specified,
    fixed length, taken from a binomial distribution.
    The parameters are defined by the model.

    Returns:
        Memory: A tuple containing a numpy array of binary codes and it's
        corresponding polarities.
    """
    code_array = get_binary_codes(
        mu = memory_size,
        m = code_length,
        distribution = distribution,
        *args,
        **kwargs
    )
    polarity_array = polarity(x = code_array)

    memory = Memory(
        codes = code_array,
        polarities = polarity_array
    )

    return memory


def get_binary_codes(
    mu: int,
    m: int,
    distribution: str = "binomial",
    *args,
    **kwargs
) -> np.ndarray:
    """
    Return a list of size "mu" of random binary codes of length "m" taken from
    a specified distribution.

    Args:
        mu (int): Number of binary codes to be generated.
        m (int): Code's length.
        distribution (str, optional): Distribution to sample from.
            Supported distributions are "binomial", "poisson", and "from_list".
            Defaults to "binomial".
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: An array of binary codes.

    Raises:
        AssertionError: If the specified distribution requires additional
        arguments and they are not provided.

    Examples:
        >>> get_binary_codes(10, 5)
        array([[1, 0, 1, 0, 1],
               [0, 1, 0, 1, 1],
               [1, 1, 0, 0, 1],
               [0, 0, 1, 1, 0],
               [1, 1, 1, 0, 0],
               [0, 1, 1, 1, 0],
               [1, 0, 0, 1, 0],
               [1, 0, 1, 1, 1],
               [0, 1, 0, 0, 0],
               [1, 0, 0, 0, 1]])

    """
    if distribution == "binomial":
        numbers = np.random.binomial(2**m, 0.5, size=mu)
    elif distribution == "poisson":
        assert "lam" in kwargs.keys(), (
            "You must provide a lambda value if 'distribution' is 'poisson'."
        )
        lam = kwargs["lam"]
        numbers = np.random.poisson(lam, size=mu)
    elif distribution == "from_list":
        assert "base_list" in kwargs.keys(), (
            "You must provide a list of integers to sample from if "
            "'distribution' is 'from_list'."
        )
        base_info_list = kwargs["base_list"]
        probabilities = np.linspace(0, 1, len(base_info_list) + 1)[1:]

        def f(p: float) -> int:
            for k in range(len(probabilities)):
                if p <= probabilities[k]:
                    return base_info_list[k]

        numbers = np.random.uniform(0, 1, size=mu)
        numbers = [f(x) for x in numbers]

    code_list = [generate_code(a, m) for a in numbers]
    code_array = np.asarray(code_list)

    return code_array


def generate_code(x: int, m: int) -> np.ndarray:
    """
    Generate a binary code of length "m" for a given integer "x".

    Args:
        x (int): An integer.
        m (int): Size of the binary code (for standardization).

    Returns:
        np.ndarray: A binary code.
    """
    code = complete_zeros(to_bin(x), m)
    code = string_to_binary(code)
    return code


def probability_distribution(
    memory: Memory,
    memory_size: int
) -> CodeDistribution:
    """
    Return a probability distribution defined over a list of binary codes.

    Args:
        memory (Memory): A Memory object (array of binary codes and its
        polarities)
        memory_size (int): The size of the memory (number of codes)

    Returns:
        CodeDistribution: An numpy array with probabilities for each code
        (identified by its integer value - array index).
    """
    codes = memory.codes
    probability_distribution = np.zeros(2**code_length)

    integers = np.matmul(codes, POWERS_OF_TWO)

    for code in integers:
        probability_distribution[code] += 1
    probability_distribution /= memory_size

    code_distribution = CodeDistribution(
        distribution = {
            code: probability_distribution[k]
            for k, code in enumerate(_A.keys())
        }
    )

    return code_distribution


def complete_probability_distribution(
    incomplete_distribution: CodeDistribution
) -> CodeDistribution:
    """
    Receives a probability distribution of an individual's memory. The memory
    may or may not contain all the possible information of the Alphabet,
    hence this function creates another dictionary with all codes from the
    Alphabet and attributes the correct probabilities from the individual's
    distribution.

    Args:
        incomplete_distribution (CodeDistribution): A probability distribution
        of an individual's memory that may be incomplete.

    Returns:
        CodeDistribution: An extension of the initial distribution including
        all the possible codes from the Alphabet.
    """
    new_A = deepcopy(_A)

    distribution = incomplete_distribution.distribution

    for code in distribution:
        new_A[code] = distribution[code]

    new_distribution = CodeDistribution(
        distribution = new_A
    )

    return new_distribution


A = {n: generate_code(n, code_length) for n in range(2**code_length)}

_A = {
    binary_to_string(generate_code(n, code_length)): generate_code(
        n,
        code_length
    )
    for n in range(2**code_length)
}
