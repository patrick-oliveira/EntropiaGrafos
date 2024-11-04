import numpy as np

from copy import deepcopy
from opdynamics import code_length
from opdynamics.math.polarity import polarity
from opdynamics.utils.types import CodeDistribution, Memory
from opdynamics.components.utils import (
    generate_random_samples,
    binary_to_string,
    POWERS_OF_TWO
)


def initialize_memory(
    memory_size: int,
    code_dimensions: int = 2,
    distribution: str = "normal",
    random_seed: int = 42,
    mean: np.array = None,
    cov: float = None,
    *args,
    **kwargs
) -> Memory:
    codes = generate_random_samples(
        n_samples = memory_size,
        n_dimensions = code_dimensions,
        random_seed = random_seed,
        distribution = distribution,
        mean = mean,
        cov = cov,
    )
    polarities = polarity(codes)

    memory = Memory(
        codes = codes,
        polarities = polarities
    )

    return memory


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
