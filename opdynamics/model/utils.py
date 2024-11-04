import numpy as np

from random import sample
from typing import List, Dict


def order_indexes(
    N: int,
    polarization_type: int,
    degrees: List[int]
) -> List[int]:
    """
    Orders the indexes of individuals in a network based on their degree.

    Args:
        N (int): The number of individuals in the network.
        polarization_type (int): The type of polarization to use. 0 for
        totally random, 1 for most connected individuals, 2 for less connected
        individuals.
        degrees (List[int]): A list of degrees for each individual in the
        network.

    Returns:
        List[int]: A list of the ordered indexes of individuals in the network.
    """
    if polarization_type == 0:
        # totally random
        indices = sample(list(range(N)), k=N)
    elif polarization_type == 1:
        # most connected individuals are polarized
        indices = list(np.argsort([degrees[x] for x in range(N)]))
        indices.reverse()
    elif polarization_type == 2:
        # less connected individuals are polarized
        indices = list(np.argsort([degrees[x] for x in range(N)]))

    return indices


def group_indexes(
    indexes: List[int],
    alpha: float,
    omega: float,
    N: int,
    *args,
    **kwargs
) -> Dict[int, List[int]]:
    """
    Groups the given indexes into three categories based on their position in
    the list.

    Args:
        indexes (List[int]): The list of indexes to be grouped.
        alpha (float): The percentage of indexes to be assigned to the alpha
        group.
        omega (float): The percentage of indexes to be assigned to the omega
        group.
        N (int): The total number of indexes in the list.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Dict[int, List[int]]: A dictionary containing three keys (-1, 0, 1)
        representing the three groups,
        and their corresponding indexes as values.
    """
    idx_alpha = int(alpha * N)
    idx_omega = int(alpha * N) + int(omega * N)

    group_alpha = indexes[:idx_alpha]
    group_omega = indexes[idx_alpha:idx_omega]
    group_neutral = indexes[idx_omega:]

    groups = {
        1: group_alpha,
        -1: group_omega,
        0: group_neutral
    }

    return groups
