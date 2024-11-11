import numpy as np

from opdynamics.utils.types import Memory


def shannon_entropy(P: np.ndarray) -> float:
    P = P[P > 0]
    return - (P * np.log2(P)).sum()


def memory_entropy(memory: Memory) -> float:
    X_grid = np.linspace(
        np.min(memory["codes"]),
        np.max(memory["codes"]),
        100
    )
    X_grid = np.meshgrid(X_grid, X_grid)
    X_grid = np.stack([X_grid[0].ravel(), X_grid[1].ravel()], axis=1)

    log_density = memory["distribution"].score_samples(X_grid)
    density = np.exp(log_density)
    density /= np.sum(density)

    return shannon_entropy(density)


def JSD(memory_x: Memory, memory_y: Memory) -> float:
    grid = np.linspace(
        np.min([memory_x['codes'], memory_y['codes']]),
        np.max([memory_x['codes'], memory_y['codes']]),
        100
    )
    grid = np.meshgrid(grid, grid)
    grid = np.array(grid).T.reshape(-1, 2)

    Px = memory_x['distribution'].score_samples(grid)
    Px = np.exp(Px)
    Px /= np.sum(Px)
    Py = memory_y['distribution'].score_samples(grid)
    Py = np.exp(Py)
    Py /= np.sum(Py)

    Pm = (Px + Py) / 2

    return shannon_entropy(Pm) - (shannon_entropy(Px) + shannon_entropy(Py)) / 2 # noqa


def S(memory_x: Memory, memory_y: Memory) -> float:
    return 1 - JSD(memory_x, memory_y)
