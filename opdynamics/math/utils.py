import numpy as np

from typing import Optional


def generate_random_samples(
    n_samples: int,
    n_dimensions: int,
    distribution: str = "normal",
    seed: Optional[int] = None,
    **distribution_kwargs
) -> np.ndarray:
    """
    Generate random samples from a specified distribution.

    Parameters:
    n_samples (int): Number of samples to generate.
    n_dimensions (int): Number of dimensions for each sample.
    seed (int, optional): Seed for the random number generator.
    Defaults to None.
    distribution (str, optional): Type of distribution to sample from.
    Defaults to "normal".
        - "normal": Samples from a standard normal distribution.
        - "multivariate_normal": Samples from a multivariate normal
        distribution.
          Additional parameters:
            - mean (array-like, optional): Mean of the distribution.
            Defaults to a zero vector.
            - cov (array-like, optional): Covariance matrix of the
            distribution. Defaults to an identity matrix.
        - "uniform": Samples from a uniform distribution over [0, 1).

    **distribution_kwargs: Additional keyword arguments for the specified
    distribution.

    Returns:
    np.ndarray: An array of shape (n_samples, n_dimensions) containing the
    generated samples.

    Raises:
    ValueError: If an unknown distribution is specified.
    """
    np.random.seed(seed)
    match distribution:
        case "normal":
            loc = distribution_kwargs.get("loc", 0.0)
            scale = distribution_kwargs.get("scale", 1.0)
            return np.random.normal(
                size = (n_samples, n_dimensions),
                loc = loc,
                scale = scale
            )
        case "multivariate_normal":
            mean = distribution_kwargs.get("mean", np.zeros(n_dimensions))
            cov = distribution_kwargs.get("cov", np.eye(n_dimensions))
            return np.random.multivariate_normal(
                mean = mean,
                cov = cov,
                size = n_samples
            )
        case "uniform":
            low = distribution_kwargs.get("low", 0.0)
            high = distribution_kwargs.get("high", 1.0)
            return np.random.uniform(
                size = (n_samples, n_dimensions),
                low = low,
                high = high
            )
        case _:
            raise ValueError(f"Unknown distribution: {distribution}")
