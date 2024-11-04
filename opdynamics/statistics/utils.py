import os
import numpy as np
import pickle

from pathlib import Path
from itertools import product
from numpy import ndarray
from typing import List, Dict
from opdynamics.statistics.handler import StatisticHandler


def get_repetitions(dirpath: str) -> List[str]:
    """
    Get a list of file names for the saved statistics of each simulation
    repetition.

    Args:
        dirpath (str): The path to the directory.

    Returns:
        List[str]: A list of file names that satisfy the conditions.
    """
    return [x for x in os.listdir(dirpath) if "stats" in x and "last" not in x]


def compute_run_mean_stats(
    experiment_params: Dict[str, List[float | int]],
    results_path: str,
    T: int
) -> Dict[str, ndarray]:
    """
    Compute the mean statistics along repetitions for a given set of
    experiment parameters.

    Args:
        experiment_params (Dict[str, List[float | int]]): A dictionary
        containing the experiment parameters.
        results_path (str): The path to the directory containing the results
        of the experiments.
        T (int): The number of repetitions.

    Returns:
        Dict[str, ndarray]: A dictionary containing the mean statistics for
        each combination of experiment parameters.
    """
    mean_stats: Dict[str, ndarray] = {}
    handler = StatisticHandler()

    for param in product(*experiment_params.values()):
        input_path = Path(results_path) / str(param)
        try:
            runs = get_repetitions(input_path)
        except Exception:
            # insert logging
            continue

        handler.load_from_files(runs)
        mean_stats[param] = handler.mean_along_repetitions()


def error_curves(
    results_path: str,
    T: int,
    *args,
    **kwargs
) -> Dict[str, List[float]]:

    # accumulates the summation of the statistics along repetitions
    entropy_sum = np.zeros(T)
    proximity_sum = np.zeros(T)
    polarity_sum = np.zeros(T)

    # saves the previous and current mean curves,
    # in order to compute their absolute difference
    (mean_entropy_i,
     mean_entropy_f) = (np.zeros(T),
                        np.zeros(T))
    (mean_proximity_i,
     mean_proximity_f) = (np.zeros(T),
                          np.zeros(T))
    (mean_polarity_i,
     mean_polarity_f) = (np.zeros(T),
                         np.zeros(T))

    # accumulates the mean absolute differences
    entropy_abs_dif = []
    proximity_abs_dif = []
    polarity_abs_dif = []

    runs = get_repetitions(results_path)
    for k, run in enumerate(runs):
        # counting runs
        n = k + 1

        try:
            stats = pickle.load(open(results_path / run, "rb"))
        except Exception as e:
            print(f'Error loadind stats: {results_path / run}')
            raise e

        # unpacking the statistics of the current run
        entropy = stats['entropy']
        proximity = stats['proximity']
        polarity = stats['polarity']

        # updating the summation of the statistics
        entropy_sum += entropy
        proximity_sum += proximity
        polarity_sum += polarity

        # computing new mean curves
        mean_entropy_f = entropy_sum / n
        mean_proximity_f = proximity_sum / n
        mean_polarity_f = polarity_sum / n

        # computing the mean absolute difference in relation to the
        # previous statistics
        entropy_abs_mean_difference = ((mean_entropy_f
                                        - mean_entropy_i)**2).mean()
        proximity_abs_mean_difference = ((mean_proximity_f
                                          - mean_proximity_i)**2).mean()
        polarity_abs_mean_difference = ((mean_polarity_f
                                         - mean_polarity_i)**2).mean()

        # saves the mean absolute difference
        entropy_abs_dif.append(entropy_abs_mean_difference)
        proximity_abs_dif.append(proximity_abs_mean_difference)
        polarity_abs_dif.append(polarity_abs_mean_difference)

        # updating the previous mean curves
        mean_entropy_i = mean_entropy_f
        mean_proximity_i = mean_proximity_f
        mean_polarity_i = mean_polarity_f

    return {
        "entropy": entropy_abs_dif,
        "proximity": proximity_abs_dif,
        "polarity": polarity_abs_dif
    }
