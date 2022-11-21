import os
import pickle
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import List, Tuple

import numpy as np

from opdynamics.utils.tools import param_to_hash
from opdynamics.utils.types import Dict, Parameters


def get_runs(path: str):
    return [x for x in os.listdir(path) if "run" in x and "last" not in x]

def get_mean_stats(param_list: Dict, results_path: str, T: int) -> Dict:
    mean_stats = {}
        
    for param in product(*param_list.values()):
        input_path = Path(results_path) / str(param)
        try:
            runs = get_runs(input_path)
        except:
            continue
        
        mean_run_stats = {
            "Entropy": np.zeros(T),
            "Proximity": np.zeros(T),
            "Polarity": np.zeros(T),
            "Distribution": np.zeros((32, T))
        }
        
        num_runs = len(runs)
        
        for run in runs:
            stats = pickle.load(open(input_path / run, "rb"))
            mean_run_stats['Entropy'] += stats['Entropy']
            mean_run_stats['Proximity'] += stats['Proximity']
            mean_run_stats['Polarity'] += stats['Polarity']
            mean_run_stats['Distribution'] += np.array(stats['Distribution']).T
            
        mean_run_stats['Entropy'] /= num_runs
        mean_run_stats['Proximity'] /= num_runs
        mean_run_stats['Polarity'] /= num_runs
        mean_run_stats['Distribution'] /= num_runs
        
        mean_stats[param] = mean_run_stats
        
    return mean_stats

def error_curve(
    results_path: str,
    T: int,
) -> Dict[str, List[float]]:
    entropy_abs_dif   = []
    proximity_abs_dif = []
    polarity_abs_dif  = []
    
    (mean_entropy_i,
     mean_entropy_f)   = (np.zeros(T),
                          np.zeros(T))
    (mean_proximity_i,
     mean_proximity_f) = (np.zeros(T),
                          np.zeros(T))
    
    (mean_polarity_i,
     mean_polarity_f)  = (np.zeros(T),
                          np.zeros(T))

    entropy_sum   = np.zeros(T)
    proximity_sum = np.zeros(T)
    polarity_sum  = np.zeros(T)
    
    runs = get_runs(results_path)
    
    for k, run in enumerate(runs):
        n = k + 1
        
        try:
            stats = pickle.load(open(results_path / run, "rb"))
        except Exception as e:
            print(f'Error loadind stats: {results_path / run}')
            raise e
        
        entropy   = stats['Entropy']
        proximity = stats['Proximity']
        polarity  = stats['Polarity']
        
        entropy_sum   += entropy
        proximity_sum += proximity
        polarity_sum  += polarity
        
        
        mean_entropy_f   = entropy_sum / n
        mean_proximity_f = proximity_sum / n
        mean_polarity_f  = polarity_sum / n
        
        
        entropy_abs_mean_difference   = ((mean_entropy_f - mean_entropy_i)**2).mean()
        proximity_abs_mean_difference = ((mean_proximity_f - mean_proximity_i)**2).mean()
        polarity_abs_mean_difference  = ((mean_polarity_f - mean_polarity_i)**2).mean()  
              
        # print("Mean Absolute Difference between runs {} and {}".format(n, n - 1))
        # print("{0: <34}: {:0.10f}".format("Entropy Mean Absolute Difference", entropy_abs_mean_difference))
        # print("{0: <34}: {:0.10f}".format("Proximity Mean Absolute Difference", proximity_abs_mean_difference))
        # print("{0: <34}: {:0.10f}".format("Polarity Mean Absolute Difference", polarity_abs_mean_difference))
        
        entropy_abs_dif.append(entropy_abs_mean_difference)
        proximity_abs_dif.append(proximity_abs_mean_difference)
        polarity_abs_dif.append(polarity_abs_mean_difference)
        
        mean_entropy_i = mean_entropy_f
        mean_proximity_i = mean_proximity_f
        mean_polarity_i = mean_polarity_f
        
    return {
        "entropy": entropy_abs_dif,
        "proximity": proximity_abs_dif,
        "polarity": polarity_abs_dif
    }


class Statistic:
    '''
    
    '''
    def compute(self, model):
        '''
        

        Parameters
        ----------
        model : Model
            DESCRIPTION.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        raise NotImplementedError
        
    def get_rep_mean(self, statistics: np.array) -> np.array:
        raise NotImplementedError
        
class StatisticHandler:
    '''
    
    '''
    def __init__(self):
        '''
        

        Returns
        -------
        None.

        '''
        self.stats_definitions = {}
        self.stats_values      = {}
        self.repetitions = []

    def new_statistic(self, name: str, function: Statistic) -> None:
        '''
        

        Parameters
        ----------
        name : str
            DESCRIPTION.
        function : Statistic
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        '''
        if not (name in self.stats_definitions.keys()):
            self.stats_definitions[name] = function
            self.stats_values[name] = []
    
    def update_statistics(self, model) -> None:
        '''
        

        Parameters
        ----------
        model : Model
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        '''
        for statistic in self.stats_definitions.keys():
            self.stats_values[statistic].append(self.stats_definitions[statistic].compute(model))
            
    def save_statistics(self, file_name: str = "statistics") -> None:
        '''
        

        Parameters
        ----------
        file_name : str, optional
            DESCRIPTION. The default is "statistics".

        Returns
        -------
        None
            DESCRIPTION.

        '''
        pickle.dump(self.stats_values, open(file_name+".pickle", "rb"))
        
    def get_statistics(self, rep_stats: bool = False) -> Dict:
        if rep_stats:
            return self.repetitions
        else:
            return self.stats_values
    
    def reset_statistics(self, hard_reset: bool = False) -> None:
        if hard_reset:
            self.stats_definitions = {}
            self.stats_values      = {}
            self.repetitions       = []
        else:
            for statistic in self.stats_definitions.keys():
                self.stats_values[statistic] = []
    
    def end_repetition(self) -> None:
        self.repetitions.append(deepcopy(self.stats_values))
        self.reset_statistics()
        
    def get_rep_mean(self) -> Dict:
        assert len(self.repetitions) > 0, 'At least one repetition should be completed to extract mean values of statistics.'
        mean_statistics = {}
        for statistic in self.stats_definitions.keys():
            stats_array = np.asarray([rep_stats[statistic] for rep_stats in self.repetitions])
            mean_statistics[statistic] = self.stats_definitions[statistic].get_rep_mean(stats_array)
            
        return mean_statistics
        

class MeanEntropy(Statistic):
    def compute(self, model) -> float:
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
    
    def get_rep_mean(self, statistics: np.array) -> np.array:
        return statistics.mean(axis = 0) if len(statistics.shape) > 1 else statistics
    
class MeanPolarity(Statistic):
    def compute(self, model) -> float:
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
    
    def get_rep_mean(self, statistics: np.array) -> np.array:
        return statistics.mean(axis = 0) if len(statistics.shape) > 1 else statistics
    
class MeanProximity(Statistic):
    def compute(self, model) -> float:
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
    
    def get_rep_mean(self, statistics: np.array) -> np.array:
        return statistics.mean(axis = 0) if len(statistics.shape) > 1 else statistics
    
class MeanDelta(Statistic):
    def compute(self, model) -> float:
        
        return np.asarray([model.ind_vertex_objects[node].delta for node in model.G]).mean()
    
    def get_rep_mean(self, statistics: np.array) -> np.array:
        return statistics.mean(axis = 0) if len(statistics.shape) > 1 else statistics
    
class MeanTransmissions(Statistic):
    def compute(self, model) -> float:
        return np.asarray([(model.ind_vertex_objects[node].transmissions, model.G.degree[node]) for node in model.G])
    
    def get_rep_mean(self, statistics: np.array) -> np.array:
        return statistics.mean(axis = 0) if len(statistics.shape) > 1 else statistics
    
class MeanAcceptances(Statistic):
    def compute(self, model) -> float:
        return np.asarray([(model.ind_vertex_objects[node].acceptances, model.G.degree[node]) for node in model.G])
                          
    def get_rep_mean(self, statistics: np.array) -> np.array:
        return statistics.mean(axis = 0) if len(statistics.shape) > 1 else statistics
                          
                        
class InformationDistribution(Statistic):
    def compute(self, model) -> np.array:
        '''
        

        Parameters
        ----------
        model : Model
            DESCRIPTION.

        Returns
        -------
        probability_distribution : TYPE
            DESCRIPTION.

        '''
        P = np.asarray([model.ind_vertex_objects[node].P_array*model.mu for node in model.G]).sum(axis = 0)/(model.mu*model.N)
        return P
    
    def get_rep_mean(self, statistics: np.array) -> np.array:
        return statistics.mean(axis = 0) if len(statistics.shape) > 1 else statistics
        
