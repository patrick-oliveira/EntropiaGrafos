from copy import deepcopy

from opdynamics.components import Individual
from opdynamics.model import Model
from opdynamics.utils.tools import (build_degree_distribution,
                                    sample_from_distribution)


def add_new_node(
    model: Model,
    n_new_connections: int,
    tendency: int
) -> Model:
    new_model = deepcopy(model)
    
    degree_dist = build_degree_distribution(new_model)
    new_node_id = len(new_model.G.nodes())

    new_model.G.add_node(
        new_node_id, 
        Object = Individual(new_model.kappa, new_model.mu),
        Tendency = tendency
    )

    new_model._ind_vertex_objects = nx.get_node_attributes(new_model.G, 'Object')
    new_model._vertex_tendencies = nx.get_node_attributes(new_model.G, 'Tendency')
    for k in range(n_new_connections):
        current_neighbours = list(new_model.G.neighbors(new_node_id))
        new_neighbour = sample_from_distribution(degree_dist)
        while new_neighbour in current_neighbours:
            new_neighbour = sample_from_distribution(degree_dist)
            
        new_model.G.add_edge(new_node_id, new_neighbour)
        
    new_model.compute_model_measures()
    
    return new_model

def modify_model(model: Model) -> Model:
    model = add_new_node(model, n_new_connections = 250, tendency = 1)
    model = add_new_node(model, n_new_connections = 250, tendency = -1)
    
    return model