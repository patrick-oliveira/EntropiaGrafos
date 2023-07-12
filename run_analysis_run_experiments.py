import argparse
import json
import random
import multiprocessing as mp

from opdynamics.simulation.experimentation import build_param_list, worker
from opdynamics.utils.tools import split_list
if __name__ == "__main__":
    num_processes = 2
    
    params_paths = [
        "experiments_params/run_analysis_epsilon_4_polarized.json",
        "experiments_params/run_analysis_epsilon_5_polarized.json",
        "experiments_params/run_analysis_epsilon_6_polarized.json",
        "experiments_params/run_analysis_epsilon_7_polarized.json",
    ]
    for params_path in params_paths:
        params = json.load(open(params_path, "r"))
        simulation_params = build_param_list(params)   
        random.shuffle(simulation_params)     
        simulation_params = split_list(simulation_params, num_processes)
        worker_input = list(zip(range(1, num_processes + 1), simulation_params))
        
        print(f"Initializing simulations with {num_processes} workers.")
        print(f"Number of iterations: {params['general_params']['T']}")
        print(f"Number of repetitions: {params['general_params']['num_repetitions']}")
        
        with mp.Pool(processes = num_processes) as pool:
            pool.map(worker, worker_input)