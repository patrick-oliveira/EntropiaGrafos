import numpy as np
from copy import deepcopy
from Scripts.Types import Memory, Binary, CodeDistribution
from Scripts.Parameters import m
from Scripts.Polarity import polarity


def initialize_memory(mu: int, m: int) -> Memory:
    '''E
    Input:
        mu: Number of codes.
        m : Code length.
        
    Return a list of size 'mu' of random binary codes of length 'm' taken from a Binomial distribution of parameters (2**m, 0.5). This list corresponds to 
    '''
    return get_binary_codes(mu = mu, m = m) 


def get_binary_codes(mu: int, m: int) -> Memory:
    '''
    Input:
        mu: Number of codes.
        m : Code length.
        
    Return a list of binary codes (without prefix) of integers drawn from a Binomial Distribution with parameters (2**m, 0.5)
    '''
    return [generate_code(a, m) for a in np.random.binomial(2**m, 0.5, size = mu)] if mu != None else \
           [generate_code(np.random.binomial(2**m, 0.5), m)]
           
def generate_code(x: int, m: int) -> Binary:
    code = complete_zeros(to_bin(x), m)
    return (code, polarity(code))


def to_bin(x: int) -> Binary:
    '''
    Input:
        x: A integer.
    
    Return the binary code of "x", without prefix.
    '''
    return bin(x)[2:]


def to_int(x: Binary) -> int:
    '''
    Input:
        x: A binary code without prefix.
        
    Convert the binary code "x" to a integer.
    '''
    return int('0b'+x, 2)

def complete_zeros(x: Binary, m: int) -> Binary:
    '''
    Input:
        x: A binary code without prefix.
        m: The desired number of bits.
        
    Complete the number of bits in 'x' with zeroes. It may be necessary to create a list of binary codes with the same length.
    '''
    return '0'*(m - len(x))+x

def probability_distribution(codes: Memory) -> CodeDistribution:
    '''
    Input:
        codes: A list of binary codes.
        
    Return a probability distribution determined over a list of binary codes.
    '''
    return complete_probability_distribution({code[0]:(lambda x: codes.count(x)/(len(codes)))(code) for code in set(codes)})
    
    
def complete_probability_distribution(incomplete_distribution: CodeDistribution) -> CodeDistribution:
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
            return code
        
A = {generate_code(x, m)[0]:0 for x in range(2**m)}             # Alphabet (work on that later)

# %%
