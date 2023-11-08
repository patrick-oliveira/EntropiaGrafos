import numpy as np

from typing import Dict, Tuple
from opdynamics import CODE_LENGTH
from opdynamics.math.polarity import polarity
from opdynamics.utils.types import (Binary,
                                    CodeDistribution,
                                    Memory,
                                    Polarity)
from opdynamics.components.utils import (to_bin,
                                         complete_zeros,
                                         string_to_binary,
                                         binary_to_string,
                                         POWERS_OF_TWO)


def initialize_memory(
    memory_size: int,
    code_length: int,
    distribution: str,
    seed: int,
    *args,
    **kwargs
) -> Tuple[Memory, Polarity]:
    """
    Create a list of size "mu" of random binary codes of an specified, fixed
    length, taken from a binomial distribution. The parameters are defined by
    the model.

    Returns:
        Memory: A tuple containing a numpy array of binary codes and it's
        corresponding polarities.
    """
    code_array = get_binary_codes(
        mu=memory_size,
        m=code_length,
        distribution=distribution,
        seed=seed,
        *args,
        **kwargs
    )
    polarity_array = polarity(code_array, code_length, seed)
    return code_array, polarity_array


def get_binary_codes(
    mu: int,
    m: int,
    distribution: str,
    seed: int,
    *args,
    **kwargs
) -> Memory:
    """
    Return a list of size "mu" of random binary codes of length "m" taken from
    a Binomial distribution of parameters (2**m, 0.5).

    Args:
        mu (int): Number of binary codes to be generated.
        m (int): Code's length.

    Returns:
        Memory: A np.array of binary codes.
    """
    np.random.seed(seed)

    if distribution == "binomial":
        integers = np.random.binomial(2**m, 0.5, size=mu)

    elif distribution == "poisson":
        lam = kwargs["lam"] if "lam" in kwargs.keys() else 1
        integers = np.random.poisson(lam, size=mu)

    elif distribution == "from_list":
        assert "base_list" in kwargs.keys(), \
            ("You must provide a list of integers to sample from if "
             "'distribution' is 'from_list'.")

        base_info_list = kwargs["base_list"]
        probabilities = np.linspace(0, 1, len(base_info_list) + 1)[1:]

        def f(p: float) -> int:
            for k in range(len(probabilities)):
                if p <= probabilities[k]:
                    return base_info_list[k]
        integers = np.random.uniform(0, 1, size=mu)
        integers = [f(x) for x in integers]

    return np.asarray([generate_code(a, m) for a in integers])


def generate_code(x: int, m: int) -> Binary:
    """
    Generate a binary code of length "m" for a given integer "x".

    Args:
        x (int): An integer.
        m (int): Size of the binary code (for standartization).

    Returns:
        Binary: A binary code.
    """
    code = complete_zeros(to_bin(x), m)
    code = string_to_binary(code)
    return code


def probability_distribution(
    memory: Memory,
    memory_size: int,
    code_length: int,
    *args,
    **kwargs
) -> CodeDistribution:
    """
    Calculates the probability distribution of the codes in the memory.

    Args:
        memory (Memory): The memory to calculate the probability distribution.
        memory_size (int): The size of the memory.

    Returns:
        CodeDistribution: A dictionary with the probability of each code.
    """
    # Calculate de frequency of each code
    distribution = np.zeros(2**code_length)
    integers = np.matmul(memory, POWERS_OF_TWO)
    for code in integers:
        distribution[code] += 1
    distribution /= memory_size

    return {code: distribution[k]
            for k, code in enumerate(_A().keys())}


def A() -> Dict[int, Binary]:
    return {n: generate_code(n, CODE_LENGTH) for n in range(2**CODE_LENGTH)}


def _A() -> Dict[str, Binary]:
    """
    Generates a dictionary with binary strings as keys and their corresponding
    generated codes as values.

    Returns:
    dict: A dictionary with binary strings as keys and their corresponding
    generated codes as values.
    """
    return {
        binary_to_string(
            generate_code(
                n,
                CODE_LENGTH
            )
        ): generate_code(
            n,
            CODE_LENGTH
        )
        for n in range(2**CODE_LENGTH)
    }
