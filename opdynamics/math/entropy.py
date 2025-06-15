import numpy as np

from opdynamics.utils.types import Memory


def shannon_entropy(P: np.ndarray) -> float:
    mask = P > 0
    P = P[mask]
    return -(P * np.log2(P)).sum()


def memory_entropy(memory: Memory) -> float:
    log_density = memory["distribution"].score_samples(memory["codes"])
    density = np.exp(log_density)
    density /= density.sum()

    return shannon_entropy(density)


def JSD(memory_x: Memory, memory_y: Memory) -> float:
    # Compute min/max once
    codes = np.concatenate([memory_x['codes'], memory_y['codes']], axis=0)
    Px = memory_x['distribution'].score_samples(codes)
    Px = np.exp(Px)
    Px /= np.sum(Px)
    Py = memory_y['distribution'].score_samples(codes)
    Py = np.exp(Py)
    Py /= np.sum(Py)

    Pm = (Px + Py) / 2

    return shannon_entropy(Pm) - (shannon_entropy(Px) + shannon_entropy(Py)) / 2 # noqa


def S(memory_x: Memory, memory_y: Memory) -> float:
    return 1 - JSD(memory_x, memory_y)
