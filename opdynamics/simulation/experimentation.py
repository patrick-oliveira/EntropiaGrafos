import os
import pickle
from pathlib import Path
from time import time
from typing import List, Tuple

from opdynamics.model import Model
from opdynamics.simulation.utils import get_param_tuple
from opdynamics.simulation.simulation import evaluate_model, initialize_model
from opdynamics.utils.reading_tools import param_to_hash
from opdynamics.utils.types import Parameters


def worker(worker_input: Tuple[int, List[Parameters]]):
    worker_id  = worker_input[0]
    param_list = worker_input[1]

    if not isinstance(param_list, list):
        param_list = [param_list]

    num_params = len(param_list)
    print(
        f"[WORKER {worker_id}] Starting processes. "
        f"Number of parameters combinations to simulate: {num_params}"
    )

    param_o_p = param_list[0]["general_parameters"]["results_path"]

    for k, params in enumerate(param_list):
        print(f"[WORKER {worker_id}] Param set {k + 1} out of {num_params}")

        simulation_params_tuple = get_param_tuple(params)
        params_hash = param_to_hash(simulation_params_tuple)
        output_path = Path(param_o_p) / params_hash

        new_experiment = not output_path.exists()
        if new_experiment:
            print(
                f"[WORKER {worker_id}] No previous runs found. "
                "Creating new model."
            )
            model = make_new_experiment(params, output_path)
            last_run = -1
        else:
            model, last_run = load_experiment(output_path)

            if last_run == -2 or last_run == int(params['num_repetitions']):
                print(
                    f"[WORKER {worker_id}] This parameter combination has "
                    f"already been simulated up to {params['num_repetitions']}"
                    " or up to an acceptable error threshold."
                )
                continue
            else:
                print(
                    f"[WORKER {worker_id}] Loaded existing model. "
                    f"Last run: {last_run}."
                )

        print(f"[WORKER {worker_id}] Starting simulation.")
        start = time()

        general_params = params["general_parameters"]

        try:
            elapsed_time, statistic_handler = evaluate_model(
                model,
                T = general_params["T"],
                num_repetitions = general_params["num_repetitions"],
                early_stop = general_params["early_stop"],
                epsilon = general_params["epsilon"],
                last_run = last_run,
                verbose = True,
                save_runs = True,
                save_path = output_path,
                worker_id = worker_id,
            )
        except Exception as e:
            print(output_path)
            raise e

        end = time()
        print(
            f"[WORKER {worker_id}] Finished simulation. "
            f"Elapsed time (s): {end - start}"
        )


def make_new_experiment(
    params: Parameters,
    output_path: Path,
    **kwargs
) -> Model:
    os.makedirs(output_path)
    model = initialize_model(
        **params["simulation_parameters"],
        **params["general_parameters"],
        **kwargs
    )

    with open(output_path / "initial_model.pkl", "wb") as file:
        pickle.dump(model, file)

    f = open(output_path / "last_run.txt", "w")
    f.write("-1")
    f.close()

    return model


def load_experiment(
    output_path: Path
) -> Tuple[Model, int]:
    f = open(output_path / "last_run.txt", "r")
    last_run = int(f.read())
    f.close()
    model = pickle.load(open(output_path / "initial_model.pkl", "rb"))

    return model, last_run
