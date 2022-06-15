from scripts.Model import initialize_model, evaluateModel
from itertools import product
from pathlib import Path
from typing import Tuple 
from multiprocessing import Pool
from time import time
import pickle 
import numpy as np
import os

def worker(params: Tuple[int]) -> Tuple[Tuple[int], dict]:
    graph_type, N, mu, code_length, kappa,\
        lambd, alpha, omega, gamma, prefferential_att = params

    identifier = (graph_type, N, mu, code_length, kappa, lambd, alpha, omega, gamma, prefferential_att,
                  parameters['T'], parameters['num_repetitions'], parameters['seed'])
    
    print(f"Identifier: {identifier}")
    
    outputPath = parameters['path_str'] / str(identifier)
    if not outputPath.exists():
        print("No previous runs found. Creating new model.")
        os.makedirs(outputPath)
        model = initialize_model(graph_type = graph_type, N = N, memory_size = mu, code_length = code_length, kappa = kappa, lambd = lambd, alpha = alpha, omega = omega, gamma = gamma, seed = parameters['seed'], prefferential_att = prefferential_att)
        num_repetitions = parameters['num_repetitions']
        T = parameters['T']
        last_run = 0
    else:
        print("Found previous runs. Loading model.")
        model = pickle.load(open(outputPath / "model.pkl", "rb"))
        
        runs = os.listdir(outputPath)
        runs = [x for x in runs if "run" in x]
        runs = [x.split("_")[1] for x in runs]
        runs = [int(x) for x in runs]
        last_run = max(runs)
        
        num_repetitions = parameters['num_repetitions'] - last_run
        T = parameters['T']
        
    print(f"Starting simulation.")
    start = time()
    elapsedTime, rep_statistics, mean_statistics = evaluateModel(model, T, num_repetitions, last_run = last_run, verbose = True, save_runs = True, save_path = outputPath)
    print(f"Finished simulation of model with parameters tuple: {params} \t - \t Execution time: {time() - start} s")
    
    return (params, mean_statistics, rep_statistics, elapsedTime)
        
    

if __name__ == "__main__":
    parameters = {
        'graph_type': ['barabasi'],
        'network_size':  [1000],
        'memory_size': [160],
        'code_length': [5],
        'kappa': [0, 15, 30],
        'lambda': [0],
        'alpha': [0],
        'omega': [0],
        'gamma': [-3, 0, 3],
        'T': 5000,
        'num_repetitions': 200,
        'seed': 42,
        'prefferential_att': [2],
        'path_str': Path("experiments/experiment_15/")
    }
    
    parameters['path_str'].mkdir(parents = True, exist_ok = True)
    with open(parameters['path_str'] / 'description.txt', 'w') as file:
        file.write("Simulations of the model without polarization, variying gamma and kappa, but keeping the others parameters fixed. Computes the information distribution.\n")
        file.write("Parameters:\n")
        file.write(str(parameters))
    
    params_cartesian_product = product(parameters['graph_type'], parameters['network_size'], parameters['memory_size'], parameters['code_length'], 
                                       parameters['kappa'], parameters['lambda'], 
                                       parameters['alpha'], parameters['omega'], parameters['gamma'], parameters['prefferential_att'])
    params_cartesian_product = list(params_cartesian_product)
    mean_results_dictionary = {}
    rep_results_dictionary = {}
    
    print(f"Initializing simulations with {parameters['T']} iterations, for {parameters['num_repetitions']} repetitions.")
    
    with Pool() as pool:
        result = pool.map(worker, params_cartesian_product)
        
    for identifier, mean_statistics, rep_statistics, elapsedTime in result:
        mean_results_dictionary[identifier] = mean_statistics
        rep_results_dictionary[identifier]  = rep_statistics
    
    pickle.dump(mean_results_dictionary, open(parameters['path_str'] / "simulation_mean_results.pickle", "wb"))  
    pickle.dump(rep_results_dictionary, open(parameters['path_str'] / "simulation_rep_results.pickle", "wb"))    
    
    partial_results = [x for x in os.listdir(parameters['path_str']) if 'partial_results' in x]  
    for pr in partial_results:
        os.remove(parameters['path_str'] / pr)
    