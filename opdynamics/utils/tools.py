import hashlib
import numpy as np
import os
import pickle

from itertools import product
from opdynamics.model import Model
from opdynamics.utils.types import Parameters
from itertools import islice
from typing import Dict, List

def validate_params(params: Parameters) -> bool:
    if params["alpha"] + params["omega"] > 1:
        return False
    
    return True

def param_to_hash(params: tuple) -> str:
    param_tuple = params
    string = str(param_tuple).encode("utf-8")
    return str(hashlib.sha256(string).hexdigest())

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


def count_runs(param_list: dict, experiments_path: str) -> int:
    params = list(product(*param_list.values()))
    params = [x[:-5] for x in params if validate_params(make_dict(x))]
    
    d = {}
    for param in params:
        run_path = get_path(param, experiments_path)
        d[param] = len(get_runs(run_path))
    
    return d

def get_mean_stats(param_list: dict, experiment_path: str, T: int) -> dict:
    mean_stats = {}
    
    params = list(product(*param_list.values()))
    params = [x[:-5] for x in params if validate_params(make_dict(x))]
    for param in params:
        run_path = get_path(param, experiment_path)
        mean_stats[param] = get_mean_run_stats(run_path, T)
        
    return mean_stats

def split_list(input_list: List, number_of_slices: int) -> List[List]:
    '''
    Create a list of slices from 'input_list' of fixed size.
    
    If L = len(input_list) and n = number_of_slices, then the Division Algorithm says that there are integers q and r < n such that L = q*n + r. If L is divisible by n, then the output list will be a set of n slices of size q, otherwise, there will be r slices of size q + 1 and and n - r slices of size q.
    
    Args:
        input_list (List): List to be segmented into sublists of potentially equal size.
        segment_size (Int): Number of slices.
        
    Output:
        output_segments (List[List]): A list of slices of fixed size 
    '''
    L = len(input_list)
    n = number_of_slices
    input_list = iter(input_list)
    
    if L// n != 0:
        split_groups_sizes = [L//n]*n
        for i in range(L%n):
            split_groups_sizes[i] = split_groups_sizes[i] + 1
    else:  
        split_groups_sizes = [L]
    
    output_segments = [list(islice(input_list, size)) for size in split_groups_sizes]
    
    return output_segments if len(output_segments) != 1 else output_segments[0]

def sort_dict(x: dict) -> dict:
    return {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse = False)}

def sample_from_distribution(dist: Dict[object, float]) -> object:
    x = np.random.uniform()
    res = None
    for k in dist.keys():
        if x < dist[k]:
            res = k
            break
    return res

def build_degree_distribution(model: Model) -> Dict[int, float]:
    degree_sum = sum(dict(model.G.degree()).values())
    
    degree_dist = {k: model.G.degree()[k]/degree_sum for k in dict(model.G.degree()).keys()}
    degree_dist = sort_dict(degree_dist)
    a = [(k, degree_dist[k]) for k in degree_dist.keys()]
    for k in range(len(degree_dist)):
        degree_dist[a[k][0]] = sum([a[r][1] for r in range(k + 1)])
        
    return degree_dist