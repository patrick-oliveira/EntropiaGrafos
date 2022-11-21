import argparse
import json
import multiprocessing as mp
import os
import pickle
from itertools import product
from pathlib import Path
from time import time
from typing import List, Tuple

import numpy as np

from opdynamics.model import Model
from opdynamics.simulation import evaluate_model
from opdynamics.simulation.experimentation import (load_experiment,
                                                   make_new_experiment)
from opdynamics.utils.tools import param_to_hash, split_list
from opdynamics.utils.types import Parameters

parser = argparse.ArgumentParser()    

def worker(worker_input: Tuple[int, List[Parameters]]):
    worker_id  = worker_input[0]
    param_list = worker_input[1]
    num_params = len(param_list)
    print(f"[WORKER {worker_id}] Starting processes. Number of parameters combinations to simulate: {num_params}")
    
    for k, params in enumerate(param_list):
        print(f"[WORKER {worker_id}] Param set {k + 1} out of {num_params}")
        output_path = Path(params["results_path"]) / param_to_hash(params)
        
        new_experiment = not output_path.exists()
        if new_experiment:
            print(f"[WORKER {worker_id}] No previous runs found. Creating new model.")  
            model = make_new_experiment(params, output_path)
        else:
            model, last_run = load_experiment(output_path)
            
            if last_run == -2:
                print(f"[WORKER {worker_id}] This parameter combination has already been simulated up to {params['num_repetitions']} or up to an acceptable error threshold.")
                continue
            else:
                print(f"[WORKER {worker_id}] Loaded existing model. Last run: {last_run}.")
            
        print(f"[WORKER {worker_id}] Starting simulation.")
        start = time()
        
        elapsed_time, statistic_handler = evaluate_model(
            model, 
            last_run = last_run, 
            verbose = True, 
            save_runs = True, 
            save_path = output_path,
            worker_id = worker_id,
            **params, 
        )
        
        end = time()
        print(f"[WORKER {worker_id}] Finished simulation. Elapsed time (s): {end - start}")


if __name__ == "__main__":
    parser.add_argument(
        "-params_path", 
        action = 'store', 
        required = True, 
        dest = 'params_path',
        help = 'Endereço para um arquivo json contendo os parâmetros da simulação'
    )
    parser.add_argument(
        "--num_processes",
        action = 'store',
        required = False,
        default = -1,
        type = int,
        dest = 'num_processes',
        help = "Número de processos a serem carregados para paralelizar a simulação de múltiplas combinações de parâmetros."
    )
    
    arguments = parser.parse_args()
    params = json.load(open(arguments.params_path, "r"))
    num_processes = mp.cpu_count() if arguments.num_processes == -1 else arguments['num_processes']
    
    simulation_params = params['simulation_params']
    simulation_params_keys = list(params['simulation_params'].keys())
    general_params = params['general_params']
    
    if not Path(general_params["experiment_path"]).exists():
        os.makedirs(general_params["experiment_path"])
    
    simulation_params = list(product(*simulation_params.values()))
    
    for k, _params in enumerate(simulation_params):
        params = {k:val for k, val in zip(simulation_params_keys, _params)}
        if params['alpha'] + params['omega'] > 1:
            simulation_params.pop(k)
            
    simulation_params = split_list(simulation_params, num_processes)
    worker_input = list(zip(range(1, num_processes + 1), simulation_params))
    
    print(f"Initializing simulations with {num_processes} workers.")
    print(f"Number of iterations: {general_params['T']}")
    print(f"Number of repetitions: {general_params['num_repetitions']}")
    
    with mp.Pool(processes = num_processes) as pool:
        pool.map(worker, worker_input)