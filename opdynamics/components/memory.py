import numpy as np

from sklearn.neighbors import KernelDensity  # type: ignore
from opdynamics.utils.types import Memory, KernelDensityEstimator
from opdynamics.math.utils import generate_random_samples


def initialize_memory(
    memory_size: int,
    info_dimension: int = 2,
    distribution: str = "normal",
    random_seed: int = 42,
    *args,
    **kwargs
) -> Memory:
    """
    Initialize the memory with random samples, polarities, and a density
    estimator.

    Args:
        memory_size (int): The number of samples to generate for the memory.
        info_dimension (int, optional): The dimensionality of the information.
        Defaults to 2.
        distribution (str, optional): The type of distribution to use for
        generating random samples. Defaults to "normal".
        random_seed (int, optional): The seed for the random number generator.
        Defaults to 42.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments to pass to the random sample
        generator and density estimator.
    Returns:
        Memory: An instance of the Memory class containing the generated codes,
        polarities, and density estimator.
    """
    codes = generate_random_samples(
        n_samples = memory_size,
        n_dimensions = info_dimension,
        random_seed = random_seed,
        distribution = distribution,
        **kwargs
    )
    polarities = polarity(codes)
    density_estimator = probability_distribution(codes, **kwargs)

    return Memory(
        codes = codes,
        polarities = polarities,
        distribution = density_estimator
    )


def polarity(x: np.ndarray, *args, **kwargs) -> np.ndarray:
    return np.array([0.5 for _ in range(x.shape[0])])


def probability_distribution(
    codes: np.ndarray,
    kernel: str = "gaussian",
    bandwidth: float = 1.0,
    find_best_bandwidth: bool = False,
    *args,
    **kwargs
) -> KernelDensityEstimator:
    """
    Estimate the probability density function of the given data using Kernel
    Density Estimation (KDE).

    Params:
    codes (np.ndarray): Input data for which the probability density function
    is to be estimated.
    kernel (str, optional): The kernel to use for the KDE.
    Default is 'gaussian'.
    bandwidth (float, optional): The bandwidth of the kernel. Default is 1.0.
    find_best_bandwidth (bool, optional): If True, automatically find the best
    bandwidth. Default is False.
    *args: Additional arguments to pass to the KernelDensity estimator.
    **kwargs: Additional keyword arguments to pass to the KernelDensity
    estimator.

    Returns:
    KernelProbabilityDensity: The fitted KernelDensity estimator.

    Raises:
    NotImplementedError: If find_best_bandwidth is True, as automatic
    bandwidth selection is not implemented yet.
    """
    if find_best_bandwidth:
        raise NotImplementedError("Automatic bandwidth selection is not implemented yet.") # noqa

    kde = KernelDensity(
        kernel = kernel,
        bandwidth = bandwidth
    )

    kde.fit(codes)

    return kde
