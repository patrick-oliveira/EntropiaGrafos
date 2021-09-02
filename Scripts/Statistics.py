from Scripts.Model import Model
from typing import Dict, Callable
from Scripts.Memory import binary_to_string
import pickle

class Statistic:
    def __call__(self, model: Model) -> float:
        raise NotImplementedError
        
class StatisticHandler:
    def __init__(self):
        self.stats_definitions = {}
        self.stats_values      = {}

    def new_statistic(self, name: str, function: Statistic) -> None:
        if not (name in self.stats_definitions.keys()):
            self.stats_definitions[name] = function
            self.stats_values[name] = []
    
    def update_statistics(self, model: Model) -> None:
        for statistic in self.stats_definitions.keys():
            self.stats_values[statistic].append(self.stats_definitions[statistic](model))
            
    def save_statistics(self, file_name: str = "statistics") -> None:
        pickle.dump(self.stats_values, open(file_name+".pickle", "rb"))

class MeanEntropy(Statistic):
    def __call__(self, model: Model) -> float:
        '''
        

        Parameters
        ----------
        model : Model
            DESCRIPTION.

        Returns
        -------
        float
            DESCRIPTION.

        '''
        return model.H
    
class MeanPolarity(Statistic):
    def __call__(self, model: Model) -> float:
        '''
            

        Parameters
        ----------
        model : Model
            DESCRIPTION.

        Returns
        -------
        float
            DESCRIPTION.

        '''
        return model.pi
    
class MeanProximity(Statistic):
    def __call__(self, model: Model) -> float:
        '''
        

        Parameters
        ----------
        model : Model
            DESCRIPTION.

        Returns
        -------
        float
            DESCRIPTION.

        '''
        return model.J

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