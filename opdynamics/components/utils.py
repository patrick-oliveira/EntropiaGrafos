import numpy as np


def generate_random_samples(
    n_samples: int,
    n_dimensions: int,
    seed: int = None,
    distribution: str = "normal",
    **distribution_kwargs
) -> np.ndarray:
    np.random.seed(seed)
    match distribution:
        case "normal":
            return np.random.normal(size=(n_samples, n_dimensions))
        case "multivariate_normal":
            mean = distribution_kwargs.get("mean", np.zeros(n_dimensions))
            cov = distribution_kwargs.get("cov", np.eye(n_dimensions))
            return np.random.multivariate_normal(mean, cov, n_samples)
        case "uniform":
            return np.random.uniform(size=(n_samples, n_dimensions))
        case _:
            raise ValueError(f"Unknown distribution: {distribution}")
