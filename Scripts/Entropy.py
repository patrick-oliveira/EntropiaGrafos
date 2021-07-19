import numpy as np
from Scripts.Types import Memory, CodeDistribution, Binary
from Scripts.Parameters import m
from functools import partial

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
    # In the first case, memory is a list of (Binary, code)
    # In the second, memory is a list of Binary
    # This can be rewriten
    entropy = (lambda x: distribution[x[0]]*np.log2(distribution[x[0]]) if distribution[x[0]] != 0 else 0) if type(memory[0]) != str else \
              (lambda x: distribution[x]*np.log2(distribution[x]) if distribution[x] != 0 else 0)
    

    return - sum(map(entropy, list(set(memory))))


# def JSD(Pu: CodeDistribution, Pv: CodeDistribution) -> float:
#     '''
#     Input:
#         Pu: A probability distribution over a list of binary codes.
#         Pv: A probability distribution over a list of binary codes.
    
#     Return the Jensen-Shannon Divergence between two probabilities distributions, Pu and Pv.
#     '''
#     return empirical_entropy(list(Pu.keys()), {code:(Pu[code] + Pv[code])/2 for code in Pu}) - \
#             (empirical_entropy(list(Pu.keys()), Pu) + empirical_entropy(list(Pv.keys()), Pv))/2
            
def JSD(Pu: CodeDistribution, Pv: CodeDistribution) -> float:
    M = {code:(Pu[code] + Pv[code])/2 for code in Pu}
    
    return (D(Pu, M) + D(Pv, M))/2

def D(P: CodeDistribution, Q: CodeDistribution) -> float:
    '''
    Input:
        P: A probability distribution over a list of binary codes.
        Q: A probability distribution over a list of binary codes.
        
    Return the Kullback-Leibler Divergence between two probability distributions.
    '''
    # return sum(map(relative_information, P.keys()))
    return sum(map(partial(_D, P, Q), P.keys()))

def _D(P: CodeDistribution, Q: CodeDistribution, x = Binary):
    if P[x] == 0:
        return 0
    else:
        if Q[x] == 0:
            return np.inf
        else:
            return P[x]*np.log2(P[x]/Q[x])
