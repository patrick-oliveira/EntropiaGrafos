from scripts.Model import initialize_model, evaluateModel
from itertools import product
from pathlib import Path
from typing import Tuple 
from multiprocessing import Pool
from time import time
import pickle 

def worker(params: Tuple[int]) -> Tuple[Tuple[int], dict]:
    N, mu, prefferential_att, code_length, kappa,\
                      gamma, lambd, alpha, omega = params

    identifier = (N, mu, prefferential_att, code_length, kappa, gamma, lambd, alpha, omega,
                  parameters['T'], parameters['num_repetitions'], parameters['seed'])

    print(f"Simulating model with parameters tuple: {params}")
    start = time()
    model = initialize_model(N, prefferential_att, mu, code_length, kappa, lambd, alpha, omega, gamma, parameters['seed'])
    elapsedTime, rep_statistics, mean_statistics = evaluateModel(model, parameters['T'], parameters['num_repetitions'])
    print(f"Finished simulation of model with parameters tuple: {params} \t - \t Execution time: {time() - start} s")
    
    return (params, mean_statistics, elapsedTime)
        
    

if __name__ == "__main__":
    parameters = {
        'network_size':  [1500],
        'memory_size': [100],
        'prefferential_att': [2],
        'code_length': [5],
        'kappa':[0, 5, 10, 15],
        'gamma': [-7, -5, -3, -1, 0, 1, 3, 5, 7],
        'lambda': [0],
        'alpha': [0],
        'omega': [0],
        'T': 150,
        'num_repetitions': 5,
        'seed': 42,
        'path_str': Path("experiments/experiment_2/")
    }
    
    parameters['path_str'].mkdir(parents = True, exist_ok = True)
    with open(parameters['path_str'] / 'description.txt', 'w') as file:
        file.write("Simulations of the model without polarization, variying gamma and kappa, but keeping the others parameters fixed.\n")
        file.write("Parameters:\n")
        file.write(str(parameters))
    
    params_cartesian_product = product(parameters['network_size'], parameters['memory_size'], 
                                       parameters['prefferential_att'], parameters['code_length'], 
                                       parameters['kappa'], parameters['gamma'], parameters['lambda'], 
                                       parameters['alpha'], parameters['omega'])
    params_cartesian_product = list(params_cartesian_product)
    results_dictionary = {}
    
    print(f"Initializing simulations with {parameters['T']} iterations, for {parameters['num_repetitions']} repetitions.")
    
    with Pool() as pool:
        result = pool.map(worker, params_cartesian_product)
        
    for identifier, statistics, elapsedTime in result:
        results_dictionary[identifier] = (statistics, elapsedTime)
    
    pickle.dump(results_dictionary, open(parameters['path_str'] / "simulation_results.pickle", "wb"))       
    