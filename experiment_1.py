from Scripts.Model import initialize_model, evaluateModel
from itertools import product
from copy import deepcopy
from pathlib import Path
import pickle 

if __name__ == "__main__":
#     _N = [500, 1000, 1500, 2000, 2500, 3000]
    prefferential_att = 2
#     _memory_size = [50, 100, 150, 200, 250, 300]
    _N = [3000]
    _memory_size = [250, 300]
    code_length = 5
    
#     kappa = 1
    gamma = 0
    lambd = 0
    alpha = 0
    omega = 0
    
    seed = 42
    
    T = 150
    num_repetitions = 5
    
    product_N_mu = product(_N, _memory_size)
    statistics_varying_N_mu = {}
    statistics_varying_kappa = {}
    for N, memory_size in product_N_mu:
        for kappa in [0, 5, 10]:
            model = initialize_model(N, prefferential_att, memory_size, code_length, kappa, lambd, alpha, omega, gamma, seed)
            _, _, mean_statistics = evaluateModel(model, T, num_repetitions = num_repetitions) 
            statistics_varying_kappa[kappa] = deepcopy(mean_statistics)
        
        statistics_varying_N_mu[f"{N}, {memory_size}"] = deepcopy(statistics_varying_kappa)
        
        statistics_path_str = "Experiments/Experiment 1/"
        statistics_path = Path(statistics_path_str)
        statistics_path.mkdir(parents = True, exist_ok = True)
        pickle.dump(statistics_varying_N_mu, open(statistics_path / f"{N}_{memory_size}.pickle", "wb")) 
                                                  
    