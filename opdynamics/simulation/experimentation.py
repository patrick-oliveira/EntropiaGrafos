import os
import pickle
from pathlib import Path
from typing import Tuple

from opdynamics.model import Model
from opdynamics.simulation import initialize_model
from opdynamics.utils.types import Parameters


def run_count(run: int, path: Path):
    f = open(path / "last_run.txt", "w")
    f.write(str(run))
    f.close()
    
def make_new_experiment(params: Parameters, output_path: Path) -> Model:
    os.makedirs(output_path)
    model = initialize_model(**params)

    with open(output_path / "initial_model.pkl", "wb") as file:
        pickle.dump(model, file)
    
    f = open(output_path / "last_run.txt", "w")
    f.write("-1")
    f.close() 
    
    return model

def load_experiment(output_path: Path) -> Tuple[Model, int]:
    f = open(output_path / "last_run.txt", "r")
    last_run = int(f.read())
    f.close()
    model = pickle.load(open(output_path / "initial_model.pkl", "rb"))
    
    return model, last_run