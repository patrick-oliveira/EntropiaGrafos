import argparse
import json
import random
import multiprocessing as mp

from opdynamics.simulation.experimentation import build_param_list, worker
from opdynamics.utils.tools import split_list

parser = argparse.ArgumentParser()    

if __name__ == "__main__":
    parser.add_argument(
        "-pp",
        "--params_path",
        type = str, 
        action = 'store', 
        required = True, 
        dest = 'params_path',
        help = 'Endereço para um arquivo json contendo os parâmetros da simulação'
    )
    parser.add_argument(
        "-np",
        "--num_processes",
        action = 'store',
        required = False,
        default = -1,
        type = int,
        dest = 'num_processes',
        help = "Número de processos a serem carregados para paralelizar a simulação de múltiplas combinações de parâmetros."
    )
    
    arguments = parser.parse_args()
    num_processes = mp.cpu_count() if arguments.num_processes == -1 else arguments.num_processes
    
    params = json.load(open(arguments.params_path, "r"))
    simulation_params = build_param_list(params)   
    random.shuffle(simulation_params)     
    simulation_params = split_list(simulation_params, num_processes)
    worker_input = list(zip(range(1, num_processes + 1), simulation_params))
    
    print(f"Initializing simulations with {num_processes} workers.")
    print(f"Number of iterations: {params['general_params']['T']}")
    print(f"Number of repetitions: {params['general_params']['num_repetitions']}")
    
    with mp.Pool(processes = num_processes) as pool:
        pool.map(worker, worker_input)