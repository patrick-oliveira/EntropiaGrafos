import numpy as np
from copy import deepcopy
from Scripts.Types import Memory, Binary, CodeDistribution
from Scripts.Parameters import seed, code_length, memory_size
from Scripts.Polarity import polarity


def initialize_memory() -> Memory:
    """
    Create a list of size "mu" of random binary codes of an specified, fixed length, taken from a binomial distribution.
    The parameters are defined by the model.
    
    Returns:
        Memory: A tuple containing a numpy array of binary codes and it's corresponding polarities.
    """   
    code_array = get_binary_codes(memory_size, code_length)
    polarity_array = polarity(code_array)
    return [code_array, polarity_array]

def get_binary_codes(mu: int, m: int) -> Memory: # Troque esse tipo de saída.
    """
    Return a list of size "mu" of random binary codes of length "m" taken from a Binomial distribution of parameters (2**m, 0.5). 

    Args:
        mu (int): Number of binary codes to be generated.
        m (int): Code's length.

    Returns:
        Memory: A np.array of binary codes.
    """    
    code_list = [generate_code(a, m) for a in np.random.binomial(2**m, 0.5, size = mu)] if mu != None else \
                [generate_code(np.random.binomial(2**m, 0.5), m)]
    code_array = np.asarray(code_list)
    return code_array
           
def generate_code(x: int, m: int) -> Binary:
    """Generate a binary code of length "m" for a given integer "x".

    Args:
        x (int): An integer.
        m (int): Size of the binary code (for standartization).

    Returns:
        Binary: A binary code.
    """         
    code = complete_zeros(to_bin(x), m)
    code = string_to_binary(code)
    return code


def to_bin(x: int) -> Binary:
    """Return the binary code of an integer "x", without prefix.

    Args:
        x (int): An integer.

    Returns:
        Binary: A binary code.
    """
    return bin(x)[2:]

def to_int(x: Binary) -> int:
    """
    Convert the binary code "x" to its correspondent integer.

    Args:
        x (Binary): A binary code.

    Returns:
        int: An integer.
    """ 
    return int('0b'+x, 2)

def complete_zeros(x: Binary, m: int) -> Binary:
    """
    Complete the number of bits in "x" with zeroes. 
    This procedure may be necessary in order to create a list of binary codes with the same length.

    Args:
        x (Binary): A binary code.
        m (int): Size of the final binary code.

    Returns:
        Binary: A binary code
    """
    return '0'*(m - len(x))+x if (m - len(x)) >= 0 else \
           x

def binary_to_string(x: Binary) -> str:
    return ''.join(list(x.astype(str))) 

def string_to_binary(x: str) -> Binary:
    return np.asarray(list(x)).astype(int)

def probability_distribution(memory: Memory) -> CodeDistribution:
    """
    Return a probability distribution defined over a list of binary codes.

    Args:
        memory (Memory): A Memory object (array of binary codes and its polarities)

    Returns:
        CodeDistribution: A dictionary with probabilities for each code.
    """    
    
    unique_codes, num_ocurrencies = np.unique(memory[0], axis = 0, return_counts = True)
    incomplete_probability_distribution = {binary_to_string(code) : num/memory_size  for code, num in zip(unique_codes, num_ocurrencies)}
    
    return complete_probability_distribution(incomplete_probability_distribution)
    
def complete_probability_distribution(incomplete_distribution: CodeDistribution) -> CodeDistribution:
    """
    Receives a probability distribution of an individual's memory. The memory may or may not countain all the possible informations of the Alphabet, 
    hence this function creates another dictionary with all codes from the Alphabet and attributes the correct probabilities from the individual's distribution.

    Args:
        incomplete_distribution (CodeDistribution): A probability distribution of an individual's memory that may be incomplete.

    Returns:
        CodeDistribution: An extension of the initial distribution including all the possible codes from the Alphabet.
    """    
    new_A = deepcopy(A)
    
    for code in incomplete_distribution:
        new_A[code] = incomplete_distribution[code]
    
    return new_A  

def random_selection(distribution: CodeDistribution) -> Binary:
    '''
    Input:
        distribution: A probability distribution over a list of binary codes.
        
    Select randomly a binary code from a given distribution.
    '''
    x = np.random.uniform()
    cumulative_probability = 0
    for code in distribution.keys():
        cumulative_probability += distribution[code]
        if cumulative_probability >= x:
            return np.asarray(list(code)).astype(int)

A = {binary_to_string(generate_code(x, code_length)):0 for x in range(2**code_length)}             # Alphabet (work on that later)