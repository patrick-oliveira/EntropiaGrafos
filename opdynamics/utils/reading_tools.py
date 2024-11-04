import json
import hashlib
import os

from pathlib import Path
from typing import List, Tuple
from opdynamics.simulation.utils import build_param_list
from opdynamics.utils.types import (
    Parameters,
    SimulationParameters
)


def parse_experiment_params(experiments_params_path: str) -> List[Parameters]:
    experiment_params = json.load(open(experiments_params_path, "r"))
    experiment_params = build_param_list(experiment_params)
    experiment_params = [x["simulation_parameters"] for x in experiment_params]

    return experiment_params


def make_tuple(
    params: Parameters,
    convert_list_values: bool = False,
) -> Tuple:
    if convert_list_values:
        return tuple(
            [tuple(x) if isinstance(x, list) else x for x in params.values()]
        )
    else:
        return tuple(params.values())


def validate_params(params: Parameters) -> bool:
    if params["alpha"] + params["omega"] > 1:
        return False

    return True


def param_to_hash(params: tuple) -> str:
    string = str(params).encode("utf-8")
    return str(hashlib.sha256(string).hexdigest())


def get_results_path(
    params: SimulationParameters,
    results_path: str
) -> str:
    param_hash = param_to_hash(make_tuple(params))
    result_path = str(Path(results_path) / param_hash)

    return result_path


def get_runs_paths(
    params: SimulationParameters,
    results_path: str
) -> List[str]:
    result_path = get_results_path(params, results_path)
    return [
        str(Path(result_path) / x)
        for x in os.listdir(result_path) if "pkl" in x and "run" in x
    ]


def count_params_runs(params: Parameters, results_path: str) -> int:
    return len(get_runs_paths(params, results_path))


def count_experiment_runs(param_list: List[Parameters], results_path):
    return {
        make_tuple(params, True): count_params_runs(params, results_path)
        for params in param_list
    }
