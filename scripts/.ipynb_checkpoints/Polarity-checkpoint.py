import numpy as np
from Scripts.Types import Weights, Binary
from Scripts.Parameters import m

def polarity_weights() -> Weights:
    '''
    Return a normalized list of 'm' random values. These values are used as weights to calculate the polarity of a binary code. 
    '''
    v = abs(np.random.randn(m))
    return v/sum(v)

def polarity(x: LBinary) -> float:
    '''
    Input: 
        X: A binary code.
        
    Return the weighted sum of bits using "beta" as weight vector.
    '''
    
    return sum([x[i]*beta[i] for i in range(m)])

beta = polarity_weights()