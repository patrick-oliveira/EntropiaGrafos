import numpy as np


def polarity(x: np.ndarray, *args, **kwargs) -> float:
    return np.array([0.5 for _ in range(x.shape[0])])
