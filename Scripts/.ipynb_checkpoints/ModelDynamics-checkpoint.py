import numpy as np
from time import time
from Scripts.Types import Graph, TransitionProbabilities, Binary, List, Dict
from Scripts.Individual import Individual
from Scripts.Entropy import JSD
from Scripts.Parameters import N, pa, mu, m


def evaluateModel(T: int, statistics: Dict,
                  kappa: float, lambd: float,
                  alpha: float, omega: float,
                  gamma: float,
                  seed: int = 42):
    '''
    Input:
    
    Evaluate a new model over T iterations.
    '''
    model = Model(N, pa, mu, m, kappa, lambd, alpha, omega, gamma, seed)
    update_statistics(model, statistics)
    
    elapsedTime = 0
    for i in range(T):
        elapsedTime += simulate(model)
        update_statistics(model, statistics)
        
    return elapsedTime

def simulate(M):
    '''
    Input:
        M: A population model.
        
    Execute one iteration of the dissemination model, updating the model's parameters at the end. Return the execution time (minutes).
    '''
    start = time()
    for u, v in M.G.edges():
        u_ind = _indInfo(M.G, u)
        v_ind = _indInfo(M.G, v)
        u_ind.receive_information(evaluate_information(distort(v_ind.X, v_ind.DistortionProbability), M.get_acceptance_probability(u, v)))
        v_ind.receive_information(evaluate_information(distort(u_ind.X, u_ind.DistortionProbability), M.get_acceptance_probability(v, u)))
    end = time()
    M.update_model()
    return (end - start)/60
    
def update_statistics(M, statistics: Dict):
    '''
    Input:
        M: A population model.
        statistics: A dictionary to accumulate statistics computed from "M"
    '''
    statistics['H'].append(M.H)
    statistics['pi'].append(M.pi)
    
def evaluate_information(code: LBinary, acceptance_probability: float):
    '''
    Input:
        code: A binary code (list).
        acceptance_probability: The probability that "code" will be accepted.
    '''
    return code if (np.random.uniform() <= acceptance_probability) else None

def _indInfo(G: Graph, node: int) -> Individual:
    '''
    Input:
        G: A population graph.
        node: An individual ID.
        
    Return the 'Individual' object from a given node.
    '''
    return G.nodes[node]['Object']

def _indTendency(G: Graph, node: int) -> str:
    '''
    Input:
        G: A population graph.
        node: An individual ID.
        
    Return the tendency of a given node.
    '''
    return G.nodes[node]['Tendency']

def get_transition_probability(ind: Individual, tendency: str) -> TransitionProbabilities:
    '''
    Inputs:
        ind: An 'Individual' object.
        tendency: "Up", "Down" or "-"
        
    Returns a dictionary with probabilities of bit distortion, i.e. the probability of 0 -> 1 and 1 -> 0, accordingly to the individual's tendency.
    '''
    return {'0': ind.delta + ind.xi, '1':ind.delta} if tendency == 'Up' else \
          ({'0': ind.delta, '1':ind.delta + ind.xi} if tendency == 'Down' else \
           {'0': ind.delta, '1':ind.delta})
          
def distort(code: LBinary, transition_probability: TransitionProbabilities) -> LBinary:
    '''
    Input:
        code: A binary code (list).
        transition_probability: Probabilities for the bit transitions 0 -> 1 and 1 -> 0.
        
    Return 'code' after bitwise distortion according to 'transition_probability'.
    '''
    
    for i in range(len(code)):
        code[i] = mutate(code[i], transition_probability[code[i]])
    
    return code


def mutate(bit: int, probability: float) -> str:
    '''
    Input:
        bit: A bit.
        probability: Distortion probability
        
    With a probability defined by "probability", distorts bit to 1 if bit = 0 and 0 if bit = 1.
    '''
    x = np.random.uniform()
    if x <= probability:
        return 1 if bit == 0 else 0
    return bit

# def to_string(bit_list: List[str]) -> str:
#     '''
#     Input:
#         character_list: A list of bits (str)

#     Transforms a list of single bits into a binary code.
#     '''
#     string = ''
#     for char in bit_list:
#         string += char
#     return string          
          
def acceptance_probability(G: Graph, u: int, v: int, gamma: float) -> float:
    '''
    Input:
        G: A population graph.
        u: An individual ID.
        v: An individual ID.
        gamma: Confidence factor
        
    Return the probability that an individual "u" will accept an information given by "v".
    '''
    u_ind = _indInfo(G, u)
    v_ind = _indInfo(G, v)
    max_sigma = max(set([u_ind.sigma**gamma]).union([_indInfo(G, w).sigma**gamma for w in G.neighbors(u)]))
    sigma_ratio = v_ind.sigma**gamma/max_sigma
    return 2/( 1/proximity(u_ind, v_ind) + 1/sigma_ratio )


def proximity(u: Individual, v:Individual) -> float:
    '''
    Input:
        u: An individual.
        v: An individual.
    
    Return the proximity between individuals u and v based on the Jensen-Shannon Divergence.
    '''
    return 1 - JSD(u.P, v.P)