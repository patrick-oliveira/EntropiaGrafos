from scripts.Model import initialize_model, evaluateModel
from itertools import product
from pathlib import Path
from typing import Tuple 
from multiprocessing import Pool
from time import time
import pickle 

def worker(params: Tuple[int]) -> Tuple[Tuple[int], dict]:
    N, mu, prefferential_att, code_length, kappa,\
    gamma, lambd, alpha, omega, T, num_repetitions, seed = params

    print(f"Simulating model with parameters tuple: {params}")
    start = time()
    model = initialize_model(N, prefferential_att, mu, code_length, kappa, lambd, alpha, omega, gamma, seed)
    _, _, mean_statistics = evaluateModel(model, T, num_repetitions)
    print(f"Finished simulation of model with parameters tuple: {params} \t - \t Execution time: {start - time()} s")
    
    return (params, mean_statistics)
        
    

if __name__ == "__main__":
    parameters = {
        'network_size':  [500, 1000],
        'memory_size': [50, 100],
        'prefferential_att': [2],
        'code_length': [5],
        'kappa': [0],
        'gamma': [0],
        'lambda': [0],
        'alpha': [0],
        'omega': [0],
        'T': 150,
        'num_repetitions': 5,
        'seed': 42,
        'path_str': Path("Experiments/Experiment 1/")
    }
    
    parameters['path_str'].mkdir(parents = True, exist_ok = True)
    with open(parameters['path_str'] / 'description.txt', 'w') as file:
        file.write("Simulations of the model without polarization, variying kappa and the size of the network, but keeping the others parameters fixed.\n")
        file.write("Parameters:\n")
        file.write(str(parameters))
    
    params_cartesian_product = product(parameters['network_size'], parameters['memory_size'], 
                                       parameters['prefferential_att'], parameters['code_length'], 
                                       parameters['kappa'], parameters['gamma'], parameters['lambda'], 
                                       parameters['alpha'], parameters['omega'], parameters['T'], 
                                       parameters['num_repetitions'], parameters['seed'])
    params_cartesian_product = list(params_cartesian_product)
    results_dictionary = {}
    
    with Pool() as pool:
        result = pool.map(worker, params_cartesian_product)
        
    for identifier, statistics in result:
        results_dictionary[identifier] = statistics
    
    pickle.dump(results_dictionary, open(parameters['path_str'] / "simulation_results.pickle", "wb"))
        
    # for N, mu, prefferential_att, code_length,\
    #     kappa, gamma, lambd, alpha, omega, T, num_repetitions, seed in params_cartesian_product:
            
    #         model = initialize_model(N, prefferential_att, mu, code_length, kappa, lambd, alpha, omega, gamma, seed)
    #         _, _, mean_statistics = evaluateModel(model, T, num_repetitions)
            
    #         identifier = (N, mu, prefferential_att, code_length, kappa, gamma, lambd, alpha, omega, T, num_repetitions, seed)
    #         results_dictionary[identifier] = mean_statistics
            
    #         # Saving partial results
            
    #         pickle.dump(results_dictionary, open(parameters['path_str'] / "simulation_results.pickle", "wb"))         
    