import os
import numpy as np
import pickle

from itertools import product
from opdynamics.utils.tools import (param_to_hash,
                                    validate_params)

def make_dict(
    param_list,
    columns = [
        'graph_type',
        'network_size',
        'memory_size',
        'code_length',
        'kappa',
        'lambd',
        'alpha',
        'omega',
        'gamma',
        'preferential_attachment',
        'polarization_grouping_type',
        'T'
    ]):
    return {k:v for k, v in zip(columns, param_list)}

def get_path(params: tuple, experiments_path: str) -> str:
    exp_path = f"{experiments_path}/{param_to_hash(params)}"
    return exp_path

def get_runs(path: str):
    return [f"{path}/{x}" for x in os.listdir(path) if "pkl" in x and "run" in x]

def get_mean_run_stats(runs_path: str, T: int) -> dict:
    try:
        runs = get_runs(runs_path)
    except Exception as e:
        raise(e)
        
    mean_run_stats = {
        "Entropy": np.zeros(T),
        "Proximity": np.zeros(T),
        "Polarity": np.zeros(T),
        "Distribution": np.zeros((32, T))
    }
    
    num_runs = len(runs)
    
    for run in runs:
        stats = pickle.load(open(run, "rb"))
        mean_run_stats['Entropy'] += stats['Entropy']
        mean_run_stats['Proximity'] += stats['Proximity']
        mean_run_stats['Polarity'] += stats['Polarity']
        mean_run_stats['Distribution'] += np.array(stats['Distribution']).T
            
    mean_run_stats['Entropy'] /= num_runs
    mean_run_stats['Proximity'] /= num_runs
    mean_run_stats['Polarity'] /= num_runs
    mean_run_stats['Distribution'] /= num_runs
    
    return mean_run_stats

def get_mean_stats(param_list: dict, experiment_path: str, T: int) -> dict:
    mean_stats = {}
    
    params = list(product(*param_list.values()))
    params = [x for x in params if validate_params(make_dict(x))]
    for param in params:
        run_path = get_path(param, experiment_path)
        mean_stats[param] = get_mean_run_stats(run_path, T)
        
    return mean_stats