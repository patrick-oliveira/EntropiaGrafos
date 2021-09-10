import pickle
import numpy as np
from Scripts.Model import Model
from Scripts.Memory import powers_of_two
from Scripts.Parameters import N, memory_size, code_length

class Statistic:
    '''
    
    '''
    def __call__(self, model: Model):
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
    
    def update_statistics(self, model: Model) -> None:
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
            self.stats_values[statistic].append(self.stats_definitions[statistic](model))
            
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


class InformationDistribution(Statistic):
    def __call__(self, model: Model) -> np.array:
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
        
