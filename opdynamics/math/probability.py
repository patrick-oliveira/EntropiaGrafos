from typing import List


def sigma(
    degree: int,
    gamma: float
) -> float:
    """
    Return the sigma value for an individual 'u'.

    Args:
        degree (int): The degree of the individual 'u'.
        gamma (float): The 'confidence factor' parameter.

    Returns:
        float: The sigma value for an individual 'u'.
    """
    return degree**gamma


def max_sigma(
    u_degree: int,
    u_neighbors_degrees: List[int],
    gamma: float,
) -> float:
    """
    Return the maximum value of sigma for an individual 'u'.

    For a given individual 'u' with degree 'd_u', sigma is defined as
    d_u^gamma, and max_sigma is defined as the maximum value of sigma
    for 'u' and its neighbors.


    Args:
        u_degree (int): The degree of the individual 'u'.
        u_neighbors_degrees (List[int]): The degrees of the neighbors of 'u'.
        gamma (float): The 'confidence factor' parameter.

    Returns:
        float: The maximum value of sigma for an individual 'u'.
    """
    return max(
        set([sigma(u_degree, gamma)]).union(
            [sigma(neighbor_degree, gamma)
                for neighbor_degree in u_neighbors_degrees]
        )
    )


def acc_prob(
    sigma_ratio: float,
    u_v_proximity: float,
):
    """
    Return the acceptance probability for an individual 'u' of informations
    given by an individual 'v'.

    Args:
        sigma_ratio (float): The ratio between the sigma value of 'v' and the
        maximum sigma value of 'u' and its neighbors.
        u_v_proximity (float): The proximity between 'u' and 'v'.

    Returns:
        float: The acceptance probability for 'u' of informations given by 'v'.
    """
    e = 1e-10
    acc_p = 2 / (1 / (u_v_proximity + e) + 1 / (sigma_ratio + e))
    return acc_p
