import pickle
import hashlib

from itertools import product
from pathlib import Path
from numpy import ndarray
from typing import Dict, List
from opdynamics.utils.types import (
    Parameters
)


def save_simulation_stats(
    stats: Dict[str, List[float | ndarray]],
    mode: str,
    stats_path: str = None,
    *args,
    **kwargs
):
    """
    Save simulation statistics.

    Args:
        stats (Dict[str, List[float | ndarray]]): A dictionary containing the
        simulation statistics.
        mode (str): The mode of saving the statistics. Currently supports
        "pickle".
        stats_path (str, optional): The path to save the statistics file.
        Required if mode is "pickle".

    Raises:
        AssertionError: If mode is "pickle" and stats_path is not provided.

    """
    if mode == "pickle":
        assert stats_path is not None, "stats_path must be provided"
        with open(stats_path, "wb") as file:
            pickle.dump(stats, file)


def check_convergence(
    error_curves: Dict[str, List[float]],
    epsilon: float,
    worker_id: int = None,
    *args,
    **kwargs
) -> bool:
    """
    Check if the error curves have converged based on the given epsilon value.

    Args:
        error_curves (Dict[str, List[float]]): A dictionary containing error
        curves for different metrics.
        epsilon (float): The convergence threshold.

    Returns:
        bool: True if all error curves have converged, False otherwise.
    """
    entropy_error = error_curves['entropy'][-1]
    proximity_error = error_curves['proximity'][-1]
    polarity_error = error_curves['polarity'][-1]

    print(
        worker_id
        + f"Errors: entropy = {entropy_error:.8f}, "
        + f"proximity = {proximity_error:.8f}, "
        + f"polarity = {polarity_error:.8f}"
    )

    converged = (
        entropy_error < epsilon
        and proximity_error < epsilon
        and polarity_error < epsilon
    )

    return converged


def run_count(run: int, path: Path):
    """
    Writes the given run number to a file named 'last_run.txt' in the
    specified path.

    Args:
        run (int): The run number to be written.
        path (Path): The path where the file will be created.

    Returns:
        None
    """
    f = open(path / "last_run.txt", "w")
    f.write(str(run))
    f.close()


def param_to_hash(params: tuple) -> str:
    """
    Transforms the list of parameters values into a hash string.

    Parameters:
        params (tuple): The dictionary containing the
        parameter values. The params must be the _simulation_params_
        dictionary, not including the general params (T, num_repetitions,
        early_stop, epsilon, and results_path)

    Returns:
        str: The hash string.
    """
    param_tuple = params
    string = str(param_tuple).encode("utf-8")

    return str(hashlib.sha256(string).hexdigest())


def validate_params(params: dict) -> bool:
    """
    Validates the parameters for a simulation.

    Args:
        params (dict): A dictionary containing the simulation parameters.

    Returns:
        bool: True if the parameters are valid, False otherwise.
    """
    if params["alpha"] + params["omega"] > 1:
        return False
    return True


def build_param_list(input_params: dict) -> List[Parameters]:
    general_params = input_params["general_params"]
    simulation_params = input_params['simulation_params']

    simulation_params_keys = input_params['simulation_params'].keys()
    # Get all combinations of parameters
    simulation_params = list(product(*simulation_params.values()))
    # Build a list of dictionaries with the keys of the original parameters
    simulation_params = list(map(
        lambda param_tuple: {
            k: val
            for k, val in zip(simulation_params_keys, param_tuple)
        },
        simulation_params
    ))
    # Add the general parameters to each valid dictionary
    simulation_params = [
        {
            "simulation_parameters": x,
            "general_parameters": general_params
        } for x in simulation_params if validate_params(x)
    ]

    return simulation_params


def get_param_tuple(params: Parameters) -> tuple:
    return tuple(params["simulation_parameters"].values())
