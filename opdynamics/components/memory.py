import numpy as np

from opdynamics.math.polarity import polarity
from sklearn.neighbors import KernelDensity
from opdynamics.utils.types import Memory, KernelProbabilityDensity
from opdynamics.components.utils import (
    generate_random_samples,
)


def initialize_memory(
    memory_size: int,
    info_dimension: int = 2,
    distribution: str = "normal",
    random_seed: int = 42,
    *args,
    **kwargs
) -> Memory:
    codes = generate_random_samples(
        n_samples = memory_size,
        n_dimensions = info_dimension,
        random_seed = random_seed,
        distribution = distribution,
        **kwargs
    )
    polarities = polarity(codes)
    distribution = probability_distribution(codes, **kwargs)

    return Memory(
        codes = codes,
        polarities = polarities,
        distribution = distribution
    )


def probability_distribution(
    codes: np.array,
    kernel: str = "gaussian",
    bandwidth: float = 1.0,
    find_best_bandwidth: bool = False,
    *args,
    **kwargs
) -> KernelProbabilityDensity:
    if find_best_bandwidth:
        raise NotImplementedError("Automatic bandwidth selection is not implemented yet.") # noqa

    kde = KernelDensity(
        kernel = kernel,
        bandwidth = bandwidth
    )

    kde.fit(codes)

    return kde
