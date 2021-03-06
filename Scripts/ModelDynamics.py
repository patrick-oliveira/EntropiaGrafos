import numpy as np
from time import time
from Scripts.Types import Graph, TransitionProbabilities, Binary, List, Dict
from Scripts.Individual import Individual
from Scripts.Entropy import JSD
from Scripts.Polarity import polarity
from Scripts.Parameters import m

    
def evaluate_information(code: Binary, acceptance_probability: float):
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
          
def distort(code: Binary, transition_probability: TransitionProbabilities) -> Binary:
    '''
    Input:
        code: A binary code (list).
        transition_probability: Probabilities for the bit transitions 0 -> 1 and 1 -> 0.
        
    Return 'code' after bitwise distortion according to 'transition_probability'.
    '''
    # This can be rewritten
    bin_list = list(code)
    
    for i in range(len(bin_list)):
        bin_list[i] = mutate(bin_list[i], transition_probability[bin_list[i]])
    
    new_code = to_string(bin_list)
    return (new_code, polarity(new_code))


def mutate(bit: int, probability: float) -> str:
    '''
    Input:
        bit: A bit.
        probability: Distortion probability
        
    With a probability defined by "probability", distorts bit to 1 if bit = 0 and 0 if bit = 1.
    '''
    x = np.random.uniform()
    if x <= probability:
        return '1' if bit == '0' else '0'
    return bit

def to_string(bit_list: List[str]) -> str:
    '''
    Input:
        character_list: A list of bits (str)

    Transforms a list of single bits into a binary code.
    '''
    string = ''
    for char in bit_list:
        string += char
    return string          
          
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