import pickle
import numpy as np
from Scripts.Memory import powers_of_two
from Scripts.Parameters import N, memory_size, code_length
from Scripts.Types import Dict
from copy import deepcopy

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
        histogram = np.arange(2**code_length)
        num_codes = N * memory_size
        
        for node in model.G:
            memory = (powers_of_two*node.L[0]).sum(axis = 1)
            for code in memory:
                histogram[code] += 1
                
        probability_distribution = histogram / num_codes
        
        return probability_distribution
        
