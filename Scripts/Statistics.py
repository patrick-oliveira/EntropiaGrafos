from Scripts.Model import Model
from typing import Dict
from Scripts.Memory import binary_to_string

def update_statistics(M: Model, statistics: Dict):
    """Updates statistics extracted from the model.

    Args:
        M (Model): A model instance.
        statistics (Dict): A dictionary with the statistics arrays to be updated.
    """ 
    statistics['H'].append(M.H)
    dist = M.compute_info_distribution()
    for code in statistics['Distribution'].keys():
        statistics['Distribution'][code].append(dist[code])
    # statistics['pi - seed = {}'.format(M.seed)].append(M.pi)
    
def compute_info_distribution(M: Model):
    hist = {info:0 for info in M.indInfo(0).P.keys()}
    N = 0
    
    for node in M.G:
        for info in M.indInfo(node).L[0]:
            info = binary_to_string(info)
            hist[info] += 1
            N += 1
    
    dist = {}
    
    for info in hist.keys():
        dist[info] = hist[info]/N
    
    return dist