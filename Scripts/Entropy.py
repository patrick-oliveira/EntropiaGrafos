__package__ = None

import numpy as np
from Scripts.Types import Memory, CodeDistribution
from Scripts.Parameters import m

# Maximum entropy
max_H = m

'''
    This can be made parallel
'''
def empirical_entropy(memory: Memory, distribution: CodeDistribution) -> float:
    '''
    Input:
        memory: A list of binary codes.
        distribution: A probability distribution definedo ver 'memory'
    '''
    entropy = lambda x: distribution[x]*np.log2(distribution[x]) if distribution[x] != 0 else 0
    return - sum(map(entropy, list(set(memory))))


def JSD(Pu: CodeDistribution, Pv: CodeDistribution) -> float:
    '''
    Input:
        Pu: A probability distribution over a list of binary codes.
        Pv: A probability distribution over a list of binary codes.
    
    Return the Jensen-Shannon Divergence between two probabilities distributions, Pu and Pv.
    '''
    return empirical_entropy(list(Pu.keys()), {code:(Pu[code] + Pv[code])/2 for code in Pu}) - \
            (empirical_entropy(Pu.keys(), Pu) + empirical_entropy(Pv.keys(), Pv))/2

def D(P: CodeDistribution, Q: CodeDistribution) -> float:
    '''
    Input:
        P: A probability distribution over a list of binary codes.
        Q: A probability distribution over a list of binary codes.
        
    Return the Kullback-Leibler Divergence between two probability distributions.
    '''
    relative_information = lambda x: P[x]*np.log2(P[x]/Q[x]) if Q[x] != 0 else (0 if P[x] == 0 else np.inf)
    sum(map(relative_information, P.keys()))
