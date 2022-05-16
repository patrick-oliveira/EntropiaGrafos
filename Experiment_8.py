from scripts.Model import initialize_model, evaluateModel
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
    start = time()
    model = initialize_model(graph_type = graph_type, N = N, memory_size = mu, code_length = code_length, kappa = kappa, lambd = lambd, alpha = alpha, omega = omega, gamma = gamma, seed = parameters['seed'])
    elapsedTime, rep_statistics, mean_statistics = evaluateModel(model, parameters['T'], parameters['num_repetitions'])
    print(f"Finished simulation of model with parameters tuple: {params} \t - \t Execution time: {time() - start} s")
    
    return (params, mean_statistics, elapsedTime)
        
    

if __name__ == "__main__":
    parameters = {
        'graph_type': ['complete'],
        'network_size':  [50],
        'memory_size': [160],
        'code_length': [5],
        'kappa': [0, 5, 10, 15, 30],
        'lambda': [0],
        'alpha': [1, 0.8, 0.6, 0.4, 0.2],
        'omega': [0, 0.2, 0.4, 0.6, 0.8],
        'gamma': [-3, 0, 3],
        'seed': 42,
        'T': 50,
        'num_repetitions': 5,
        'path_str': Path("experiments/experiment_8/")
    }
    
    parameters['path_str'].mkdir(parents = True, exist_ok = True)
    with open(parameters['path_str'] / 'description.txt', 'w') as file:
        file.write("Simulations of the model with, variying gamma and kappa, but keeping the others parameters fixed. Using a complete graph.\n")
        file.write("Parameters:\n")
        file.write(str(parameters))
    
    params_cartesian_product = product(parameters['graph_type'], parameters['network_size'], parameters['memory_size'], 
                                       parameters['code_length'], parameters['kappa'], parameters['lambda'], 
                                       parameters['alpha'], parameters['omega'], parameters['gamma'])
    params_cartesian_product = list(params_cartesian_product)
    results_dictionary = {}
    
    print(f"Initializing simulations with {parameters['T']} iterations, for {parameters['num_repetitions']} repetitions.")
    
    with Pool() as pool:
        result = pool.map(worker, params_cartesian_product)
        
    for identifier, statistics, elapsedTime in result:
        if identifier != "None" and statistics != "None" and elapsedTime != "None":
            results_dictionary[identifier] = (statistics, elapsedTime)
    
    pickle.dump(results_dictionary, open(parameters['path_str'] / "simulation_results.pickle", "wb"))       
    