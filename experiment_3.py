from Scripts.Model import initialize_model, evaluateModel
from itertools import product
from copy import deepcopy
from pathlib import Path
import pickle

# Analisar a dinâmica do sistema com polarização, variando o parâmetro lambda
# sem preferências individuais

if __name__ == "__main__":
    statistics_path_str = "Experiments/Experiment 3/"
    statistics_path = Path(statistics_path_str)
    statistics_path.mkdir(parents = True, exist_ok = True)
    
    N = 2000
    memory_size = 150
    prefferential_att = 2
    code_length = 5
    
    gamma = 0 # sem preferências individuais
    kappa_list = [5, 10, 15, 20]
    lambd_list = [0, 1, 5, 10, 15, 20, 25]
    alpha = 0
    omega = 0
    
    seed = 42
    
    T = 250
    
    num_repetitions = 10
    
    product_kappa_lambd = product(kappa_list, lambd_list)
    for kappa, lambd in product_kappa_lambd:
        model = initialize_model(N, prefferential_att, memory_size, code_length, kappa, lambd, alpha, omega, gamma, seed)
        _, _, mean_statistics = evaluateModel(model, T, num_repetitions = num_repetitions)
        pickle.dump(mean_statistics, open(statistics_path / f"{kappa}_{lambd}.pickle", "wb"))
        
    