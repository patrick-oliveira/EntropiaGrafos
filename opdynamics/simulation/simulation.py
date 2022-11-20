import pickle
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from opdynamics.model import Model
from opdynamics.model.dynamics import distort, evaluate_information
from opdynamics.model.statistics import (InformationDistribution,
                                         MeanAcceptances, MeanEntropy,
                                         MeanPolarity, MeanProximity,
                                         MeanTransmissions, StatisticHandler)


def initialize_model(
    graph_type: str, 
    network_size: int,
    memory_size: int, 
    code_length: int,
    kappa: float, 
    lambd: float,
    alpha: float, 
    omega: float,
    gamma: float,
    seed: int,
    prefferential_att: float = None, 
    degree: int = None, 
    edge_prob: float = None, 
    verbose: bool = False,
    worker_id: int = None,
    *args,
    **kwargs,
) -> Model:
    worker_id = f"[WORKER {worker_id}] " if worker_id is not None else ""
    if verbose:
        print(worker_id + "Initializing model.")
    
    start = time.time()
    initial_model = Model(
        graph_type = graph_type, 
        N = network_size, 
        mu = memory_size, 
        m = code_length, 
        kappa = kappa, 
        lambd = lambd, 
        alpha = alpha, 
        omega = omega, 
        gamma = gamma, 
        seed = seed,
        pa = prefferential_att, 
        d = degree, 
        p = edge_prob
    )
    model_initialization_time = time.time() - start
    
    if verbose:
        print(worker_id + f"Model initialized. Elapsed time: {np.round(model_initialization_time/60, 2)} min")
    
    return initial_model

def Parallel_evaluateModel(
    initial_model: Model,
    T: int, 
    num_repetitions: int, 
    verbose: bool = False
) -> Tuple[float, List[Dict], Dict]:
    pass


def evaluateModel(
    initial_model: Model,
    T: int, 
    num_repetitions: int = 1, 
    last_run: int = 0, 
    verbose: bool = False,
    save_runs: bool = False, 
    save_path: Union[Path, str] = None,
    worker_id: int = None,
) -> Tuple[float, StatisticHandler]:
    """
    Evaluate a new model over T iterations.
    """
    worker_id = f"[WORKER {worker_id}] " if worker_id is not None else ""
    if verbose:
        print(worker_id + "Model evaluation started.")
        print(worker_id + f"Number of repetitions = {num_repetitions}")
        
    simulation_time = []
    statistic_handler = StatisticHandler()
    statistic_handler.new_statistic('Entropy',   MeanEntropy())
    statistic_handler.new_statistic('Proximity', MeanProximity())
    statistic_handler.new_statistic('Polarity',  MeanPolarity())
    statistic_handler.new_statistic('Distribution', InformationDistribution())
    statistic_handler.new_statistic('Acceptance', MeanAcceptances())
    statistic_handler.new_statistic('Transmission', MeanTransmissions())
    
    
    for repetition in range(last_run + 1, num_repetitions + 1):
        if verbose:
            print(worker_id + f"Repetition {repetition}/{num_repetitions}")
        
        model = deepcopy(initial_model)
        
        start = time.time()
        for i in range(T):
            simulate(model)
            statistic_handler.update_statistics(model)
        repetition_time = time.time() - start
        simulation_time.append(repetition_time)
        
        if verbose:
            print(worker_id + f"Finished repetition {repetition}/{num_repetitions}. Elapsed time: {np.round(simulation_time[-1]/60, 2)} minutes")
        
        statistic_handler.end_repetition()
        
        if save_runs:
            with open(save_path / f"run_{repetition}_stats.pkl", "wb") as file:
                pickle.dump(statistic_handler.repetitions[-1], file)
                
        # compute error curve
        
    elapsedTime = sum(simulation_time)
    
    return elapsedTime, statistic_handler

def simulate(M: Model):
    """
    Execute one iteration of the information propagation model, updating the model's parameters at the end. 
    Return the execution time (minutes).

    Args:
        M (Model): A model instance.  

    Returns:
        None.
    """
    for u, v in M.G.edges():
        u_ind = M.indInfo(u)
        v_ind = M.indInfo(v)
        received = u_ind.receive_information(evaluate_information(distort(v_ind.X, v_ind.DistortionProbability), M.get_acceptance_probability(u, v)))
        if received:
            v_ind.transmitted()
            u_ind.received()
        received = v_ind.receive_information(evaluate_information(distort(u_ind.X, u_ind.DistortionProbability), M.get_acceptance_probability(v, u)))
        if received:
            u_ind.transmitted()
            v_ind.received()
    
    M.update_model()