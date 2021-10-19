from Scripts.Model import initialize_model, evaluateModel
from itertools import product
from copy import deepcopy
from pathlib import Path
import pickle

# Analisar a dinâmica do sistema sem polarização, variando kappa e gamma

if __name__ == "__main__":
    statistics_path_str = "Experiments/Experiment 2 v2/"
    statistics_path = Path(statistics_path_str)
    statistics_path.mkdir(parents = True, exist_ok = True)
    
    N = 1200
    memory_size = 150
    prefferential_att = 2
    code_length = 5
    
    # gamma_list = [-5, -1, 0, 1, 5]
    gamma_list = [-5, -1, 0]
    kappa_list = [0, 5, 10, 15, 20]
    lambd = 0
    alpha = 0
    omega = 0
    
    seed = 42
    
    T = 150
    num_repetitions = 200
    
    product_gamma_kappa = product(gamma_list, kappa_list)
    statistics_varying_gamma_kappa = {}
    for gamma, kappa in product_gamma_kappa:
        model = initialize_model(N, prefferential_att, memory_size, code_length, kappa, lambd, alpha, omega, gamma, seed)
        _, _, mean_statistics = evaluateModel(model, T, num_repetitions = num_repetitions)
        statistics_varying_gamma_kappa[f"{gamma}, {kappa}"] = deepcopy(mean_statistics)
        pickle.dump(statistics_varying_gamma_kappa, open(statistics_path / f"{gamma}_{kappa}.pickle", "wb"))
        
    
    