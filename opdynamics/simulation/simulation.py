import time
import numpy as np

from copy import deepcopy
from pathlib import Path
from typing import Tuple, Union
from opdynamics.model.model import Model
from opdynamics.model.dynamics import distort
from opdynamics.statistics.handler import StatisticHandler
from opdynamics.statistics.utils import error_curves
from opdynamics.simulation.utils import (save_simulation_stats,
                                         check_convergence,
                                         run_count)


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
    Evaluates a model by simulating its evolution over a given number of
    repetitions and iterations.

    Args:
        initial_model (Model): The initial model to be evaluated.
        T (int): The number of iterations to simulate the model for each
        repetition.
        num_repetitions (int, optional): The number of repetitions to perform.
        Defaults to 1.
        early_stop (bool, optional): Whether to stop the evaluation early if
        the model converges. Defaults to False.
        epsilon (float, optional): The threshold for convergence. Defaults
        to 1e-5.
        last_run (int, optional): The index of the last run. Defaults to -1.
        verbose (bool, optional): Whether to print progress messages. Defaults
        to False.
        save_runs (bool, optional): Whether to save the model stats for each
        repetition. Defaults to False.
        save_path (Union[Path, str], optional): The path to save the model
        stats. Defaults to None.
        worker_id (int, optional): The ID of the worker. Defaults to None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[float, StatisticHandler]: A tuple containing the total elapsed
        time and the statistic handler object.
    """
    worker_id = f"[WORKER {worker_id}] " if worker_id is not None else ""
    if verbose:
        print(worker_id + "Model evaluation started.")
        print(worker_id + f"Number of repetitions = {num_repetitions}")

    statistic_handler = StatisticHandler(
        statistics=[
            "entropy",
            "proximity",
            "polarity",
            # "transmissions",
            # "acceptances",
            "information_distribution"
        ]
    )
    simulation_time = []
    # start from where it has stopped
    current_run = last_run + 1
    for repetition in range(current_run, num_repetitions):
        if verbose:
            print(worker_id + f"Repetition {repetition + 1}/{num_repetitions}")

        model = deepcopy(initial_model)

        start = time.time()
        # each repetition evolves the model along T iterations
        for _ in range(T):
            simulate(model)
            statistic_handler.update_statistics(model)
        # end of repetition, taking a snapshot of the model stats
        statistic_handler.end_repetition()
        repetition_time = time.time() - start
        simulation_time.append(repetition_time)
        print(f"Elapsed time: {np.round(simulation_time[-1]/60, 2)} minutes")
        if save_runs:
            # save the model stats for each repetition,
            # this block should be agnostic regarding
            # the data structure of the stats
            file_path = save_path / f"run_{repetition + 1}_stats.pkl"
            last_rep = statistic_handler.current_rep - 1
            save_simulation_stats(
                stats=statistic_handler.repetitions[last_rep],
                mode="pickle",
                stats_path=file_path
            )

        if early_stop:
            # Check if the model has converged
            # by computing the evolution of the error curves
            # along the repetitions and see if it is below
            # the threshold epsilon
            converged = check_convergence(
                error_curves(save_path, T),
                epsilon,
                worker_id
            )
            if converged:
                run_count(-2, save_path)
                break
        run_count(repetition, save_path)

        if verbose:
            print(
                worker_id +
                f"Finished repetition {repetition + 1}/{num_repetitions}. " +
                f"Elapsed time: {np.round(simulation_time[-1]/60, 2)} minutes"
            )

    elapsedTime = sum(simulation_time)
    return elapsedTime, statistic_handler


def simulate(M: Model):
    """
    Simulates the information exchange process between nodes in a graph.

    Parameters:
    M (Model): The model containing the graph and individuals.

    Returns:
    None
    """
    # each node will exchange information with each of its neighbors
    for u, v in M.G.edges():
        u_ind = M.indInfo(u)
        v_ind = M.indInfo(v)

        # V transmits to U
        # V gets an information and distorts it according to
        # its polarization tendency and memory entropy
        v_info = v_ind.X
        v_info = distort(v_info, v_ind.DistortionProbability)

        # U has a probability of accepting the information
        # from V
        # U returns an confirmation if it accepts the information
        acceptance_v_to_u = M.get_acceptance_probability(u, v)
        received = u_ind.receive_information(
            v_info,
            acceptance_v_to_u
        )
        # If accepted, increase the transmission counter of V and
        # acceptance counter of U
        if received:
            v_ind.transmitted()
            u_ind.received()

        # U transmits to V
        # U gets an information and distorts it according to
        # its polarization tendency and memory entropy
        u_info = u_ind.X
        u_info = distort(u_info, u_ind.DistortionProbability)

        # V has a probability of accepting the information
        # from U
        # V returns an confirmation if it accepts the information
        acceptance_u_to_v = M.get_acceptance_probability(v, u)
        received = v_ind.receive_information(
            u_info,
            acceptance_u_to_v
        )
        # If accepted, increase the transmission counter of U and
        # acceptance counter of V
        if received:
            u_ind.transmitted()
            v_ind.received()

    # after all exchanges, updates the memory of each individual,
    # its statistics and the global model statistics
    M.update_model()
