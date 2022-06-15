from scripts.Model import Parallel_evaluateModel, initialize_model, evaluateModel
from itertools import product
from pathlib import Path
from typing import Tuple 
from multiprocessing import Pool
from time import time
import pickle 
import numpy as np

def worker(params: Tuple[int]) -> Tuple[Tuple[int], dict]:
    graph_type, N, mu, code_length, kappa,\
                      lambd, alpha, omega, gamma = params
                      
    if alpha + omega > 1:
        print("Invalid set of parameters alpha and omega. Returning.")
        return ("None", "None", "None")

    identifier = (graph_type, N, mu, code_length, kappa, lambd, alpha, omega, gamma,
                  parameters['T'], parameters['num_repetitions'], parameters['seed'])

    print(f"Simulating model with parameters tuple: {params}")
    print(f"Number of repetitions: {parameters['num_repetitions']}")
    start = time()
    
    model = initialize_model(graph_type = graph_type, N = N, memory_size = mu, code_length = code_length, kappa = kappa, lambd = lambd, alpha = alpha, omega = omega, gamma = gamma, seed = parameters['seed'])
    
    mean_statistics, rep_statistics = Parallel_evaluateModel(model, T = parameters['T'], num_repetitions = parameters['num_repetitions'], verbose = True)
    
    print(f"Finished simulation of model with parameters tuple: {params} \t - \t Execution time: {time() - start} s")
    
    return (params, mean_statistics, rep_statistics)
        
    

if __name__ == "__main__":
    parameters = {
        'graph_type': ['complete'],
        'network_size':  [50],
        'memory_size': [160],
        'code_length': [5],
        'kappa': [0, 15, 30],
        'lambda': [0],
        'alpha': [0],
        'omega': [0],
        'gamma': [-10, 0, 10],
        'seed': 42,
        'T': 200,
        'num_repetitions': 200,
        'path_str': Path("experiments/experiment_9/")
    }
    
    parameters['path_str'].mkdir(parents = True, exist_ok = True)
    with open(parameters['path_str'] / 'description.txt', 'w') as file:
        file.write("Simulations of the model without polarization, variying gamma and kappa, but keeping the others parameters fixed. Using a complete graph.\n")
        file.write("Parameters:\n")
        file.write(str(parameters))
    
    params_cartesian_product = product(parameters['graph_type'], parameters['network_size'], parameters['memory_size'], 
                                       parameters['code_length'], parameters['kappa'], parameters['lambda'], 
                                       parameters['alpha'], parameters['omega'], parameters['gamma'])
    params_cartesian_product = list(params_cartesian_product)
    results_dictionary = {}
    
    print(f"Initializing simulations with {parameters['T']} iterations, for {parameters['num_repetitions']} repetitions.")
    
    results = []
    for k, params in enumerate(params_cartesian_product):
        results.append(worker(params))
        
    for identifier, mean_statistics, rep_statistics in results:
        if identifier != "None" and mean_statistics != "None" and rep_statistics != "None":
            results_dictionary[identifier] = (mean_statistics, rep_statistics)
    
    pickle.dump(results_dictionary, open(parameters['path_str'] / "simulation_results.pickle", "wb"))       
    