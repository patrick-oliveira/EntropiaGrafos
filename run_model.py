import argparse
import multiprocessing as mp
import json
import logging
import os

from random import shuffle
from typing import List
from opdynamics.simulation.utils import build_param_list, get_param_tuple
from opdynamics.utils.reading_tools import param_to_hash
from opdynamics.utils.types import Parameters
from opdynamics.simulation.simulation import parallel_evaluate_model
from opdynamics.simulation.experimentation import (
    make_new_experiment,
    load_experiment
)


logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)


def load_params(params_path: str) -> List[Parameters]:
    """
    Load the parameters from a json file.

    Parameters
    ----------
    params_path : str
    """

    params = json.load(open(params_path, "r"))
    params = build_param_list(params)
    shuffle(params)

    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pp",
        "--params_path",
        type = str,
        action = 'store',
        required = True,
        dest = 'params_path',
        help =
        """
            Path for a json file containing the set of parameters.
        """
    )
    parser.add_argument(
        "-np",
        "--num_processes",
        action = 'store',
        required = False,
        default = -1,
        type = int,
        dest = 'num_processes',
        help =
        """
            Number of processes to use. If -1, the number of processes will be
            equal to the number of cores available.
        """
    )

    args = parser.parse_args()

    num_processes = (
        mp.cpu_count() if args.num_processes == -1
        else args.num_processes
    )

    param_list = load_params(args.params_path)
    for params in param_list:
        params_hash = param_to_hash(get_param_tuple(params))

        output_path = os.path.join(
            params["general_parameters"]["results_path"],
            params_hash
        )

        if not os.path.exists(output_path):
            model = make_new_experiment(params, output_path)
        else:
            model, last_run = load_experiment(output_path)

            converged = last_run == -2
            max_repetitions = (
                last_run == int(params["general_parameters"]["num_repetitions"]) # noqa
            )

            if converged or max_repetitions:
                continue

        parallel_evaluate_model(
            initial_model = model,
            T = params["general_parameters"]["T"],
            num_repetitions = params["general_parameters"]["num_repetitions"],
            num_processes = num_processes
        )
