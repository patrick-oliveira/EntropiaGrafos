import os
import pickle
from pathlib import Path
from time import time
from typing import List, Tuple

from opdynamics import SEED
from opdynamics.model.model import Model
from opdynamics.simulation.simulation import evaluate_model
from opdynamics.simulation.utils import param_to_hash
from opdynamics.utils.types import (Parameters,
                                    SimulationParameters)


def worker(worker_input: Tuple[int, List[Parameters]]):
    """
    Perform simulation tasks for a worker process.

    Args:
        worker_input (Tuple[int, List[Parameters]]): A tuple containing the
        worker ID and a list of parameter sets to simulate.

    Returns:
        None
    """
    # identifier of the worker process
    worker_id: int = worker_input[0]
    # list of parameters to simulate
    param_list = worker_input[1]

    # the worker can receive a single parameter set or a list
    # of parameter sets, so we need to make sure it is a list
    if not isinstance(param_list, list):
        param_list = [param_list]

    num_params = len(param_list)
    print(
        f"[WORKER {worker_id}] Starting processes. "
        + f"Number of parameters combinations to simulate: {num_params}"
    )

    for k, params in enumerate(param_list):
        print(f"[WORKER {worker_id}] Param set {k + 1} out of {num_params}")

        simulation_parameters = params["simulation_parameters"]
        general_parameters = params["general_parameters"]

        output_path = Path(general_parameters["results_path"])
        output_path = output_path / param_to_hash(simulation_parameters)

        # check if the experiment has already been simulated or not
        # if it has, load the model and continue the simulation from where
        # it stopped, otherwise create a new model if it is a new experiment
        # or just do nothing if it is a finished experiment
        experiment_is_new = not output_path.exists()
        if experiment_is_new:
            print(
                f"[WORKER {worker_id}] No previous runs found. "
                + "Creating new model."
            )

            model, last_run = make_new_experiment(
                simulation_parameters,
                output_path
            )

        else:
            model, last_run = load_experiment(output_path)

            num_reps = int(general_parameters["num_repetitions"])
            already_converged = last_run == -2
            finished_repetitions = last_run == num_reps
            if already_converged or finished_repetitions:
                print(
                    f"[WORKER {worker_id}] This parameter combination has "
                    + f"already been simulated up to {num_reps} or up to an "
                    + "acceptable error threshold."
                )
                continue
            else:
                print(
                    f"[WORKER {worker_id}] Loaded existing model. "
                    + f"Last run: {last_run}."
                )

        print(f"[WORKER {worker_id}] Starting simulation.")

        start = time()
        elapsed_time, statistic_handler = evaluate_model(
            initial_model=model,
            T=general_parameters["T"],
            num_repetitions=general_parameters["num_repetitions"],
            early_stop=general_parameters["early_stop"],
            epsilon=general_parameters["epsilon"],
            last_run=last_run,
            verbose=True,
            save_runs=True,
            save_path=output_path,
            worker_id=worker_id,
        )
        end = time()
        print(
            f"[WORKER {worker_id}] Finished simulation. "
            + f"Elapsed time (s): {end - start}"
        )


def make_new_experiment(
    simulation_parameters: SimulationParameters,
    output_path: Path,
    *args,
    **kwargs
) -> Model:
    """
    Creates a new experiment with the given simulation parameters and output
    path.

    Args:
        simulation_parameters (SimulationParameters): The parameters for the
        simulation.
        output_path (Path): The path where the experiment output will be saved.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[Model, int]: A tuple containing the created model and the last
        run number (-1).
    """
    model = Model(seed=SEED, **simulation_parameters)
    # Saves the initial model to a file so it can be loaded
    # for each repetition
    os.makedirs(output_path)
    with open(output_path / "initial_model.pkl", "wb") as file:
        pickle.dump(model, file)
    # Creates a file to save the last run number
    f = open(output_path / "last_run.txt", "w")
    f.write("-1")
    f.close()

    return model, -1


def load_experiment(output_path: Path) -> Tuple[Model, int]:
    """
    Load an experiment from the specified output path.

    Args:
        output_path (Path): The path to the experiment output directory.

    Returns:
        Tuple[Model, int]: A tuple containing the loaded model and the number
        of the last run.
    """
    # get the number of the last run
    f = open(output_path / "last_run.txt", "r")
    last_run = int(f.read())
    f.close()
    # load the initial model
    model = pickle.load(open(output_path / "initial_model.pkl", "rb"))

    return model, last_run
