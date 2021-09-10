import numpy as np
from Scripts.Types import Graph, TransitionProbabilities, Binary, List
from Scripts.Individual import Individual
from Scripts.Entropy import JSD
from Scripts.Polarity import polarity
from Scripts.Parameters import code_length

    
def evaluate_information(code: Binary, acceptance_probability: float) -> Binary:
    """
    Function which decides wether or not the incoming information will be accepted. If refused, return "None"

    Args:
        code (Binary): A incoming binary code.
        acceptance_probability (float): Probability of accepting "code"

    Returns:
        [Binary]: The incoming binary code, if accepted, otherwise "None".
    """   
    return code if (np.random.uniform() <= acceptance_probability) else None

def getInfo(G: Graph, node: int) -> Individual:
    """
    Return the 'Individual' object from a give vertex id.

    Args:
        G (Graph): The graph model.
        node (int): A vertex id.

    Returns:
        Individual: The 'Individual' object contained within the specified vertex.
    """    
    return G.nodes[node]['Object']

def getTendency(G: Graph, node: int) -> str:
    """Return the polarization tendency of a given node.

    Args:
        G (Graph): The graph model.
        node (int): A vertex id.

    Returns:
        str: A string which can assume the values "Up", "Down" or None.
    """
    return G.nodes[node]['Tendency']    

def get_transition_probabilities(ind: Individual, tendency:str = None) -> TransitionProbabilities:
    """
    Return a dictionary with probabilities of bit distortion, i.e. the probability of 0 -> 1 and 1 -> 0, considering the individual's tendency.

    Args:
        ind (Individual): An individual object.
        tendency (str): The identification of the individual's tendency (polarizing upwards or downwards)

    Returns:
        TransitionProbabilities: The dictionary of probabilities for the transitions 0 -> 1 and 1 -> 0.
    """    
    return {0: ind.delta + ind.xi, 1:ind.delta} if tendency == 'Up' else \
          ({0: ind.delta, 1:ind.delta + ind.xi} if tendency == 'Down' else \
           {0: ind.delta, 1:ind.delta})
          
def distort(code: Binary, transition_probability: TransitionProbabilities) -> Binary:
    """Return 'code' after bitwise distortion according to the probabilities given by "transition_probability".

    Args:
        code (Binary): A binary code.
        transition_probability (TransitionProbabilities): The probabilitions for the transitions 0 -> 1 and 1 -> 0.

    Returns:
        Binary: A possibily bitwise distorted code.
    """    
    # get the mutation probability for each bit of "code"
    transition_probabilities = np.vectorize(lambda x: transition_probability[x])(code)
    # get a vector of random numbers with same size as "code"
    random_numbers = np.random.uniform(size = code_length)
    # get a vector identifying which bits will be mutated
    mutate = (random_numbers <= transition_probabilities).astype(int)
    # get the arguments for all bits which will be mutated
    mutate = np.argwhere(mutate).reshape(-1)
    # mutate all selected bits (inverting its values)
    code[mutate] = np.logical_not(code[mutate]).astype(int)
    
    return code

def proximity(u: Individual, v:Individual) -> float:
    """
    Return the proximity between individuals u and v based on the Jensen-Shannon Divergence.

    Args:
        u (Individual): An individual 'u'
        v (Individual): An individual 'v'

    Returns:
        float: The Jensen-Shannon Divergence JSD(Pu, Pv), where Pu and Pv are the memory's probability distribution of individuals u and v, respectively.
    """    
    return 1 - JSD(u.P, v.P)

def acceptance_probability(G: Graph, u: int, v: int, gamma: float) -> float:
    """
    Return the probability that an individual "u" will accept and information given by "v".
    
    ==>> Write the latex formula here. <<==

    Args:
        G (Graph): The Graph model.
        u (int): The vertex id of an individual 'u'.
        v (int): The vertex id of an individual 'v'.
        gamma (float): The 'confidence factor' parameter.

    Returns:
        float: The acceptance probability for u of informations given by v.
    """
    u_ind = getInfo(G, u)
    v_ind = getInfo(G, v)
    
    max_sigma = max(set([len(list(G.neighbors(u)))**gamma]).union([len(list(G.neighbors(w)))**gamma for w in list(G.neighbors(u))]))
    
    sigma_ratio =(len(list(G.neighbors(v)))**gamma)/max_sigma
    return 2/( 1/proximity(u_ind, v_ind) + 1/sigma_ratio )