import argparse
import json
import random
import multiprocessing as mp

from opdynamics.simulation.experimentation import worker
from opdynamics.simulation.utils import build_param_list
from opdynamics.utils.tools import split_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pp",
        "--params_path",
        type=str,
        action='store',
        required=True,
        dest='params_path',
        help=('Endereço para um arquivo json contendo '
              + 'os parâmetros da simulação')
    )
    parser.add_argument(
        "-np",
        "--num_processes",
        action='store',
        required=False,
        default=-1,
        type=int,
        dest='num_processes',
        help=("Número de processos a serem carregados para paralelizar "
              + "a simulação de múltiplas combinações de parâmetros.")
    )
    arguments = parser.parse_args()

    if arguments.num_processes == -1:
        num_processes = mp.cpu_count()
    else:
        num_processes = arguments.num_processes

    # prepares the input for the worker processes
    params = json.load(open(arguments.params_path, "r"))
    simulation_params = build_param_list(params)
    random.shuffle(simulation_params)
    simulation_params = split_list(simulation_params, num_processes)
    worker_input = list(zip(range(1, num_processes + 1), simulation_params))

    print(f"Initializing simulations with {num_processes} workers.")
    print(f"Number of iterations: {params['general_parameters']['T']}")
    print(
        "Number of repetitions: "
        + f"{params['general_parameters']['num_repetitions']}"
    )

    with mp.Pool(processes=num_processes) as pool:
        pool.map(worker, worker_input)
