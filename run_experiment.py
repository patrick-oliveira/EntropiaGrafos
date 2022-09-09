from posixpath import split
from scripts.Model import initialize_model, evaluateModel
from scripts.utils import split_list
from itertools import product
from pathlib import Path
from typing import Tuple, List, Union
from multiprocessing import Pool
from time import time

import pickle
import numpy as np
import os
import argparse
import json
import multiprocessing as mp

parser = argparse.ArgumentParser()

def worker(worker_input: Tuple[int, List[Tuple[Union[str, int]]]]):
    worker_id  = worker_input[0]
    param_list = worker_input[1]
    
    num_params = len(param_list)
    print(f"[WORKER {worker_id}] Starting processes. Number of parameters combinations to simulate: {num_params}")
    
    for k, _params in enumerate(param_list):
        print(f"[WORKER {worker_id}] Param set {k + 1} out of {num_params}")
        params = {k:val for k, val in zip(simulation_params_keys, _params)}

        output_path = Path(general_params['experiment_path']) / str(_params)
        
        new_experiment = not output_path.exists()
        if new_experiment:
            print(f"[WORKER {worker_id}] No previous runs found. Creating new model.")
            os.makedirs(output_path)
            model = initialize_model(**params, seed = general_params['seed'])
            with open(output_path / "model.pkl", "wb") as file:
                pickle.dump(model, file)
            
            last_run = 0      
        else:
            try:
                runs = os.listdir(output_path)
                runs = [x for x in runs if "run" in x]
                runs = [x.split("_")[1] for x in runs]
                runs = [int(x) for x in runs]
                last_run = max(runs)
            except:
                last_run = 0
            if last_run == general_params['num_repetitions']:
                print(f"[WORKER {worker_id}] This parameter combination was already simulated for {general_params['num_repetitions']} repetitions.")
                continue
            
            print(f"[WORKER {worker_id}] Loading existing model. Last run: {last_run}.")
            model = pickle.load(open(output_path / "model.pkl", "rb"))
            
        print(f"[WORKER {worker_id}] Starting simulation.")
        start = time()
        
        elapsed_time, statistic_handler = evaluateModel(
            model, 
            T = general_params['T'], 
            num_repetitions = general_params['num_repetitions'], 
            last_run = last_run, 
            verbose = True, 
            save_runs = True, 
            save_path = output_path,
            worker_id = worker_id,
        )
        
        end = time()
        print(f"[WORKER {worker_id}] Finished simulation. Elapsed time (s): {end - start}")
        

excluded_list = [(0.2, 0.8), (0.4, 0.6)]


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
        if params['alpha'] + params['omega'] > 1 or (params['alpha'], params['omeag']) in excluded_list:
            simulation_params.pop(k)
            
    simulation_params = split_list(simulation_params, num_processes)
    worker_input = list(zip(range(1, num_processes + 1), simulation_params))
    
    print(f"Initializing simulations with {num_processes} workers.")
    print(f"Number of iterations: {general_params['T']}")
    print(f"Number of repetitions: {general_params['num_repetitions']}")
    
    with mp.Pool(processes = num_processes) as pool:
        pool.map(worker, worker_input)