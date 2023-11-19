import json
import hashlib
import os
import pandas as pd
import numpy as np
import pickle

from pathlib import Path
from typing import List, Tuple
from itertools import product
from opdynamics.utils.types import (Parameters,
                                    SimulationResult)

def parse_params(param_tuple: Tuple) -> Parameters:
    param_tuple = param_tuple[:-5]

    param_names = [
        "graph_type",
        "network_size",
        "memory_size",
        "code_length",
        "kappa",
        "lambd",
        "alpha",
        "omega",
        "gamma",
        "preferential_attachment",
        "polarization_type"
    ]
    
    return {m:n for m,n in zip(param_names, param_tuple)}

def parse_experiment_params(experiments_params_path: str) -> List[Parameters]:
    experiment_params = json.load(open(experiments_params_path, "r"))
    parsed_params = {}
    for k in experiment_params.keys():
        parsed_params.update(experiment_params[k])
    
    for k in parsed_params.keys():
        if type(parsed_params[k]) != list:
            parsed_params[k] = [parsed_params[k]]

    parsed_params = list(product(*parsed_params.values()))
    parsed_params = [parse_params(x) for x in parsed_params]
    parsed_params = [x for x in parsed_params if validate_params(x)]
    
    return parsed_params

def make_tuple(params: Parameters) -> Tuple:
    return tuple(params.values())

def validate_params(params: Parameters) -> bool:
    if params["alpha"] + params["omega"] > 1:
        return False
    
    return True

def get_results_path(params: Parameters, results_path: str) -> str:
    param_hash = param_to_hash(make_tuple(params))
    result_path = str(Path(results_path) / param_hash)
    
    return result_path

def get_runs_paths(params: Parameters, results_path: str) -> List[str]:
    result_path = get_results_path(params, results_path)
    return [str(Path(result_path) / x) for x in os.listdir(result_path) if "pkl" in x and "run" in x]

def count_params_runs(params: Parameters, results_path: str) -> int:
    return len(get_runs_paths(params, results_path))

def count_experiment_runs(param_list: List[Parameters], results_path):
    return {make_tuple(params): count_params_runs(params, results_path) for params in param_list}