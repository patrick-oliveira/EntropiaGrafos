import os
import pickle
from itertools import product
from pathlib import Path
from pprint import pprint
from time import time
from typing import List, Tuple

from opdynamics.model import Model
from opdynamics.simulation.simulation import evaluate_model, initialize_model
from opdynamics.utils.tools import param_to_hash
from opdynamics.utils.types import Parameters


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
            last_run = -1
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
            T = params["T"],
            num_repetitions = params["num_repetitions"],
            early_stop = params["early_stop"],
            epsilon = params["epsilon"],
            last_run = last_run, 
            verbose = True, 
            save_runs = True, 
            save_path = output_path,
            worker_id = worker_id,
        )
        
        end = time()
        print(f"[WORKER {worker_id}] Finished simulation. Elapsed time (s): {end - start}")
    
def make_new_experiment(params: Parameters, output_path: Path) -> Model:
    os.makedirs(output_path)
    model = initialize_model(**params)

    with open(output_path / "initial_model.pkl", "wb") as file:
        pickle.dump(model, file)
    
    f = open(output_path / "last_run.txt", "w")
    f.write("-1")
    f.close() 
    
    return model

def load_experiment(output_path: Path) -> Tuple[Model, int]:
    f = open(output_path / "last_run.txt", "r")
    last_run = int(f.read())
    f.close()
    model = pickle.load(open(output_path / "initial_model.pkl", "rb"))
    
    return model, last_run

def validate_params(params: dict) -> bool:
    if params["alpha"] + params["omega"] > 1:
        return False
    
    return True

def build_param_list(input_params: dict) -> List[Parameters]:
    general_params = input_params["general_params"]
    
    simulation_params_keys = input_params['simulation_params'].keys()
    simulation_params = input_params['simulation_params']
    simulation_params = list(product(*simulation_params.values()))
    simulation_params = list(map(
        lambda param_tuple: {k: val for k, val in zip(simulation_params_keys, param_tuple)},
        simulation_params
    ))
    simulation_params = [x | general_params for x in simulation_params if validate_params(x)]

    return simulation_params
    
        