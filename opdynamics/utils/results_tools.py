import pickle
import numpy as np
import pandas as pd

from typing import List
from opdynamics.utils.reading_tools import get_runs_paths, make_tuple
from opdynamics.utils.types import SimulationParameters, SimulationResult


def get_param_mean_data(
    params: SimulationParameters,
    results_path: str
) -> SimulationResult:
    runs = get_runs_paths(params, results_path)
    T = len(pickle.load(open(runs[0], "rb"))['Entropy'])
    num_runs = len(runs)

    entropy       = np.zeros(T)
    proximity     = np.zeros(T)
    polarity      = np.zeros(T)
    distribution  = np.zeros((32, T))
    acceptances   = []
    transmissions = []

    for run in runs:
        run_data = pickle.load(open(run, "rb"))

        entropy      += run_data['Entropy']
        proximity    += run_data["Proximity"]
        polarity     += run_data["Polarity"]
        try:
            distribution += np.array(run_data["Distribution"]).T
        except:
            pass

        # a = pd.DataFrame(
        #     run_data["Acceptance"][-1],
        #     columns = ["acceptances", 'degree']
        # )
        # acceptances.append(a)

        # t = pd.DataFrame(
        #     run_data["Transmission"][-1],
        #     columns = ['transmissions', 'degree']
        # )
        # transmissions.append(t)

    entropy      /= num_runs
    proximity    /= num_runs
    polarity     /= num_runs
    distribution /= num_runs

    # acceptances = pd.concat(
    #     acceptances,
    #     ignore_index = True
    # ).groupby(by = "degree").agg('mean')
    # acceptances['acceptances'] = acceptances['acceptances'].astype(int)
    # acceptances = acceptances.to_dict()["acceptances"]

    # transmissions = pd.concat(
    #     transmissions, ignore_index = True
    # ).groupby(by = "degree").agg('mean')
    # transmissions['transmissions'] = transmissions['transmissions'].astype(int)
    # transmissions = transmissions.to_dict()['transmissions']

    mean_stats = {
        "entropy": entropy,
        "proximity": proximity,
        "polarity": polarity,
        "distribution": distribution,
        # "acceptances": acceptances,
        # "transmissions": transmissions
    }

    return mean_stats


def get_experiment_mean_data(
    param_list: List[SimulationParameters],
    results_path: str
) -> dict[SimulationResult]:
    return {
        make_tuple(params, True): get_param_mean_data(params, results_path)
        for params in param_list
    }
