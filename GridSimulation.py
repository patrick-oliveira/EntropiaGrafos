from scripts.Model import initialize_model, evaluateModel
from itertools import product
from copy import deepcopy
from pathlib import Path
import pickle 

if __name__ == "__main__":
    parameters = {
        'network_size':  [500, 1000, 1500, 2000, 2500, 3000],
        'memory_size': [50, 100, 150, 200, 250, 300],
        'prefferential_att': [2],
        'code_length': [5],
        'kappa': [0, 5, 10],
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
    
    params_cartesian_product = product(parameters['network_size'], parameters['memory_size'], parameters['prefferential_att'], parameters['code_length'], 
                                       parameters['kappa'], parameters['gamma'], parameters['lambda'], parameters['alpha'], parameters['omega'])
    params_cartesian_product = list(params_cartesian_product)
    results_dictionary = {}
    for N, mu, prefferential_att, code_length,\
        kappa, gamma, lambd, alpha, omega in params_cartesian_product:
            
            model = initialize_model(N, prefferential_att, mu, code_length, kappa, lambd, alpha, omega, gamma, parameters['seed'])
            _, _, mean_statistics = evaluateModel(model, parameters['T'], parameters['num_repetitions'])
            
            identifier = (N, mu, prefferential_att, code_length, kappa, gamma, lambd, alpha, omega)
            results_dictionary[identifier] = mean_statistics
            
            # Saving partial results
            pickle.dump(results_dictionary, open(parameters['path_str'] / "simulation_results.pickle", "wb"))
                                                  
    