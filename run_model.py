import json

from time import time
from pathlib import Path
from opdynamics.utils.reading_tools import param_to_hash
from opdynamics.simulation.simulation import evaluate_model
from opdynamics.simulation.experimentation import (make_new_experiment,
                                                   load_experiment)
from argparse import ArgumentParser


parser = ArgumentParser()

if __name__ == "__main__":
    parser.add_argument(
        "--param_file",
        type=str,
        action='store',
        required=False,
        default=None,
        dest='param_file',
        help=(
            "Path for a json file containing the set of parameters. "
            "If specified, the other arguments will be ignored."
        )
    )
    parser.add_argument(
        '--graph_type',
        type=str,
        action='store',
        required=False,
        default='barabasi',
        dest='graph_type',
        help="Type of the underlying model's network."
    )
    parser.add_argument(
        '--network_size',
        type=int,
        action='store',
        required=False,
        default=500,
        dest='network_size',
        help="Number of nodes in the network"
    )
    parser.add_argument(
        '--memory_size',
        type=int,
        action='store',
        required=False,
        default=256,
        dest='memory_size',
        help="Size of the set of codes composing a node's memory."
    )
    parser.add_argument(
        '--code_length',
        type=int,
        action='store',
        required=False,
        default=5,
        dest='code_length',
        help="Number of bits in a single code."
    )
    parser.add_argument(
        '--kappa',
        type=float,
        action='store',
        required=False,
        default=None,
        dest='kappa',
        help="Distortion control coefficient."
    )
    parser.add_argument(
        '--lambda',
        type=float,
        action='store',
        required=False,
        default=None,
        dest='lambd',
        help="Polarization control coefficient."
    )
    parser.add_argument(
        '--alpha',
        type=float,
        action='store',
        required=False,
        default=None,
        dest='alpha',
        help="Proportion of the population who are positively polarized."
    )
    parser.add_argument(
        '--omega',
        type=float,
        action='store',
        required=False,
        default=None,
        dest='omega',
        help="Proportion of the population who are negatively polarized."
    )
    parser.add_argument(
        '--gamma',
        type=float,
        action='store',
        required=False,
        default=None,
        dest='gamma',
        help="Popularity bias."
    )
    parser.add_argument(
        '--preferential_attachment',
        type=int,
        action='store',
        required=False,
        default=2,
        dest='preferential_attachment',
        help=(
            "Control the number of new connection as new nodes are added "
            "to the network."
        )
    )
    parser.add_argument(
        '--polarization_type',
        type=int,
        action='store',
        required=False,
        default=0,
        dest='polarization_type',
        help="Type of node's ordering when polarizing individuals."
    )
    parser.add_argument(
        '--T',
        type=int,
        action='store',
        required=False,
        default=500,
        dest='T',
        help="Number of iterations of a model."
    )
    parser.add_argument(
        '--num_repetitions',
        type=int,
        action='store',
        required=False,
        default=2000,
        dest='num_repetitions',
        help="Number of repetitions of the experiment."
    )
    parser.add_argument(
        '--early_stop',
        action='store_true',
        dest='early_stop',
        help=(
            "Stop the replications of the experiment if the errors fall below "
            "an epsilon threshold."
        )
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        action='store',
        required=False,
        default=0.000000001,
        dest='epsilon',
        help="Convergence threshold."
    )
    parser.add_argument(
        "-np",
        "--num_processes",
        action='store',
        required=False,
        default=-1,
        type=int,
        dest='num_processes',
        help=(
            "Number of proccesses used to paralelyze the simulation of "
            "multiple parameter combinations."
        )
    )
    parser.add_argument(
        "-rp",
        "--results_path",
        action='store',
        required=False,
        default="results",
        type=str,
        dest='results_path',
        help="Path to save the results."
    )

    arguments = parser.parse_args()

    if arguments.param_file:
        params = json.load(open(arguments.param_file, "r"))
    else:
        params = {
            "graph_type": arguments.graph_type,
            "network_size": arguments.network_size,
            "memory_size": arguments.memory_size,
            "code_length": arguments.code_length,
            "kappa": arguments.kappa,
            "lambd": arguments.lambd,
            "alpha": arguments.alpha,
            "omega": arguments.omega,
            "gamma": arguments.omega,
            "preferential_attachment": arguments.preferential_attachment,
            "polarization_type": arguments.polarization_type,
            "T": arguments.T,
            "num_repetitions": arguments.num_repetitions,
            "early_stop": arguments.early_stop,
            "epsilon": arguments.epsilon
        }

    output_path = Path(arguments.results_path) / param_to_hash(
        tuple(params.values())[:-4]
    )
    new_experiment = not output_path.exists()
    if new_experiment:
        model = make_new_experiment(
            params,
            output_path,
            distribution="from_list",
            base_list=[27, 29, 31]
        )
        last_run = -1
    else:
        model, last_run = load_experiment(output_path)
        converged = last_run == -2
        all_runs = last_run == params["num_repetitions"]
        if converged or all_runs:
            print(
                "This parameter combination has already been simulated up "
                "to {} or up an acceptable error threshold.".format(
                    params["num_repetitions"]
                )
            )
        else:
            print("Loaded existing model. Last run: {}".format(last_run))

    start = time()
    elapsed_time, statistic_handler = evaluate_model(
        initial_model=model,
        last_run=last_run,
        T=params["T"],
        num_repetitions=params["num_repetitions"],
        early_stop=params["early_stop"],
        epsilon=params["epsilon"],
        verbose=True,
        save_runs=True,
        save_path=output_path,
    )
    end = time()

    print("Finished simulation. Elapsed time (s): {}".format(end - start))
