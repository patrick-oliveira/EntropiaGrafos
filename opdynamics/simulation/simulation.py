import pickle
import time
import numpy as np
import multiprocessing as mp
import logging

from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import List, Tuple, Union
from multiprocessing import Lock, Value
from opdynamics.model import Model, stats_dict
from opdynamics.model.dynamics import distort, evaluate_information
from opdynamics.model.statistics import StatisticHandler, error_curve


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
    preferential_attachment: float = None,
    polarization_grouping_type: int = 0,
    degree: int = None,
    edge_prob: float = None,
    verbose: bool = False,
    worker_id: int = None,
    distribution: str = "binomial",
    *args,
    **kwargs,
) -> Model:
    worker_id = f"[WORKER {worker_id}] " if worker_id is not None else ""
    if verbose:
        print(worker_id + "Initializing model.")

    start = time.time()
    initial_model = Model(
        graph_type = graph_type,
        network_size = network_size,
        memory_size = memory_size,
        code_length = code_length,
        kappa = kappa,
        lambd = lambd,
        alpha = alpha,
        omega = omega,
        gamma = gamma,
        preferential_attachment = preferential_attachment,
        polarization_grouping_type = polarization_grouping_type,
        d = degree,
        p = edge_prob,
        distribution = distribution,
        **kwargs
    )
    model_initialization_time = time.time() - start

    if verbose:
        print(
            worker_id +\
            "Model initialized. Elapsed time: {} min".format(
                np.round(model_initialization_time/60, 2)
            )
        )

    return initial_model

def init_statistic_handler(s_names: List[str] = None) -> StatisticHandler:
    statistic_handler = StatisticHandler()

    if s_names is None:
        s_names = [
            "Entropy",
            "Proximity",
            "Polarity",
            "Distribution",
            "Acceptance",
            "Transmission"
        ]

    for name in s_names:
        try:
            statistic_handler.new_statistic(name, stats_dict[name]())
        except Exception:
            print(f"Error building statistic: {name}")

    return statistic_handler


shared_counter = Value("i", 0)
early_stop = Value("i", 0)
LOCK = Lock()

def worker(arguments):
    if early_stop.value == 1:
            return

    logging.info("Worker started.")
    model = deepcopy(arguments.get("initial_model"))
    T = arguments.get("T")
    save_path = arguments.get("save_path")
    epsilon = arguments.get("epsilon")

    with LOCK:
        with open(save_path / "last_run.txt", "r") as f:
            last_run = int(f.read())
        if last_run == -2:
            return

    statistic_handler = init_statistic_handler()

    start = time.time()
    for i in range(T):
        simulate(model)
        statistic_handler.update_statistics(model)
    repetition_time = np.round(time.time() - start, 2)
    
    statistic_handler.end_repetition()

    with LOCK:
        if early_stop.value == 1:
                return
        repetition = shared_counter.value
        logging.info(f"Repetition ended in {repetition_time} seconds.")
        logging.info(f"Saving repetition {repetition} to {save_path}")

        with open(save_path / f"run_{repetition}_stats.pkl", "wb") as file:
            pickle.dump(statistic_handler.repetitions[-1], file)

        errors = error_curve(save_path, T)
        errors = {
            "entropy": errors["entropy"][-1],
            "proximity": errors["proximity"][-1],
            "polarity": errors["polarity"][-1]
        }

        logging.info("Last errors:")
        pprint(errors)

        if errors["entropy"] <= epsilon and \
                errors["proximity"] <= epsilon and \
                    errors["polarity"] <= epsilon:
                        logging.info(
                            "Difference between current and last runs is below "\
                            f"the {epsilon} threshold. Stopping simulation."
                        )
                        run_count(-2, save_path)
                        early_stop.value = 1
                        return
        
        run_count(repetition, save_path)
        shared_counter.value += 1


def parallel_evaluate_model(
    initial_model: Model,
    T: int,
    save_path: Path,
    num_repetitions: int = 1,
    num_processes: int = -1,
    last_run: int = -1,
    epsilon: int = None,
    *args,
    **kwargs
):
    arguments = [
        {
            "initial_model": initial_model,
            "T": T,
            "save_path": save_path,
            "epsilon": epsilon
        }
        for _ in range(num_repetitions)
    ]
    early_stop.value = 0
    shared_counter.value = last_run + 1
    logging.info(f"Last run {shared_counter.value}")

    start = time.time()
    with mp.Pool(num_processes) as pool:
        pool.map(worker, arguments)
    total_tiime = np.round(time.time() - start, 2)

    logging.info(f"Total elapsed time: {total_tiime} seconds.")

def evaluate_model(
    initial_model: Model,
    T: int,
    num_repetitions: int = 1,
    early_stop: bool = False,
    epsilon: float = 1e-5,
    last_run: int = -1,
    verbose: bool = False,
    save_runs: bool = False,
    save_path: Union[Path, str] = None,
    worker_id: int = None,
    *args,
    **kwargs
) -> Tuple[float, StatisticHandler]:
    """
    Evaluate a new model over T iterations.
    """
    worker_id = f"[WORKER {worker_id}] " if worker_id is not None else ""
    if verbose:
        logging.info(worker_id + "Model evaluation started.")
        logging.info(worker_id + f"Number of repetitions = {num_repetitions}")

    simulation_time = []
    statistic_handler = init_statistic_handler()

    current_run = last_run + 1
    for repetition in range(current_run, num_repetitions):
        if verbose:
            logging.info(worker_id + f"Repetition {repetition}/{num_repetitions}")

        model = deepcopy(initial_model)

        start = time.time()
        for _ in range(T):
            simulate(model)
            statistic_handler.update_statistics(model)
        repetition_time = time.time() - start
        simulation_time.append(repetition_time)

        if verbose:
            logging.info(
                worker_id +\
                f"Finished repetition {repetition + 1}/{num_repetitions}. "\
                f"Elapsed time: {np.round(simulation_time[-1]/60, 2)} minutes"
            )

        statistic_handler.end_repetition()

        if save_runs:
            with open(save_path / f"run_{repetition}_stats.pkl", "wb") as file:
                pickle.dump(statistic_handler.repetitions[-1], file)

        if early_stop:
            errors = error_curve(save_path, T)
            errors = {
                "entropy": errors["entropy"][-1],
                "proximity": errors["proximity"][-1],
                "polarity": errors["polarity"][-1]
            }

            logging.info(worker_id + "Last errors:")
            pprint(errors)

            if errors["entropy"] <= epsilon and \
                errors["proximity"] <= epsilon and \
                    errors["polarity"] <= epsilon:
                        logging.info(
                            worker_id +\
                            "Difference between current and last runs is below "\
                            f"the {epsilon} threshold. Stopping simulation."
                        )
                        run_count(-2, save_path)
                        break

        run_count(repetition, save_path)

    elapsedTime = sum(simulation_time)

    return elapsedTime, statistic_handler


def simulate(M: Model):
    """
    Execute one iteration of the information propagation model, updating the model's
    parameters at the end.
    Return the execution time (minutes).

    Args:
        M (Model): A model instance.

    Returns:
        None.
    """
    for u, v in M.G.edges():
        u_ind = M.indInfo(u)
        v_ind = M.indInfo(v)
        received = u_ind.receive_information(
            evaluate_information(
                distort(v_ind.X, v_ind.DistortionProbability),
                M.get_acceptance_probability(u, v)
            )
        )
        if received:
            v_ind.transmitted()
            u_ind.received()
        received = v_ind.receive_information(
            evaluate_information(
                distort(u_ind.X, u_ind.DistortionProbability),
                M.get_acceptance_probability(v, u)
            )
        )
        if received:
            u_ind.transmitted()
            v_ind.received()

    M.update_model()

def run_count(run: int, path: Path):
    f = open(path / "last_run.txt", "w")
    f.write(str(run))
    f.close()

import multiprocessing as mp
from functools import partial
import math

# The compute_edge_updates function remains largely the same,
# but it will now receive 'M' as its first argument when used with partial.
def compute_edge_updates(M, edge):
    """
    Worker function: reads state, computes outcomes, and returns proposed updates.
    It does NOT modify the model's state.
    """
    u, v = edge
    u_ind = M.indInfo(u)
    v_ind = M.indInfo(v)
    received = u_ind.receive_information(
        evaluate_information(
            distort(v_ind.X, v_ind.DistortionProbability),
            M.get_acceptance_probability(u, v)
        )
    )
    if received:
        v_ind.transmitted()
        u_ind.received()
    received = v_ind.receive_information(
        evaluate_information(
            distort(u_ind.X, u_ind.DistortionProbability),
            M.get_acceptance_probability(v, u)
        )
    )
    if received:
        u_ind.transmitted()
        v_ind.received()
        
        
def simulate_parallel_decompose(M: 'Model', num_processes: int = None):
    """
    Executes one iteration by decomposing the problem into parallel compute
    and serial update phases, using multiprocessing.Pool and functools.partial.
    """
    all_updates = []
    all_edges = list(M.G.edges()) # Get all edges as a list

    if num_processes is None:
        num_processes = mp.cpu_count() # Use all available cores by default
    elif num_processes <= 0:
        raise ValueError("num_processes must be a positive integer.")

    # Create a partial function: compute_edge_updates_with_M will now only need 'edge' as an argument
    # 'M' is "baked in" to this function.
    # When this partial function is sent to processes, 'M' is pickled and sent along.
    compute_func = partial(compute_edge_updates, M)

    # Phase 1 & 2: Parallel Compute and Gather
    with mp.Pool(processes=num_processes) as pool:
        # pool.map distributes the 'all_edges' list to the processes,
        # calling compute_func for each edge.
        # It automatically gathers all the results.
        pool.map(compute_func, all_edges)

    # Final model update is done after all state changes are applied
    M.update_model()