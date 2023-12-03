import pickle
import hashlib

from itertools import product
from pathlib import Path
from numpy import ndarray
from typing import Dict, List
from opdynamics.utils.types import (ExperimentParameters,
                                    Parameters)


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

    converged = (entropy_error < epsilon and
                 proximity_error < epsilon and
                 polarity_error < epsilon)

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
<<<<<<< HEAD
        params (tuple): The dictionary containing the
=======
        params (Dict[str, str | float | int]): The dictionary containing the
>>>>>>> 36b10b161e5083e6a0554d131dd0984d105fe07e
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


def build_param_list(input_params: ExperimentParameters) -> List[Parameters]:
    """
    Builds a list of parameter dictionaries for simulation.

    Args:
        input_params (ExperimentParameters): The input parameters for the
        experiment.

    Returns:
        List[Parameters]: A list of parameter dictionaries for simulation.

    """
    # get the experiment parameters sub-dictionaries
    simulation_params = input_params["simulation_parameters"]
    general_params = input_params["general_parameters"]

    simulation_params_keys = simulation_params.keys()
    # get all combinations of the simulation parameters
    simulation_params = list(product(*simulation_params.values()))
    # build a dictionary for each combination of parameters
    simulation_params = list(map(
        lambda param_tuple: {
            k: val
            for k, val in zip(simulation_params_keys, param_tuple)
        },
        simulation_params
    ))
    # remove invalid combinations, e.g. alpha + omega > 1
    simulation_params = [
        {
            "simulation_parameters": x,
            "general_parameters": general_params
        } for x in simulation_params if validate_params(x)
    ]

    return simulation_params
