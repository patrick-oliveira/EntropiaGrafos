import numpy as np
from Types import Memory, CodeDistribution, Binary
from Parameters import code_length, max_H
from Memory import binary_to_string
from functools import partial

def shannon_entropy(P: float) -> float:
    return P*np.log2(P)

def JSD(Pu: CodeDistribution, Pv: CodeDistribution) -> float:
    M = {code:(Pu[code] + Pv[code])/2 for code in Pu}
    
    return (D(Pu, M) + D(Pv, M))/2

def D(P: CodeDistribution, Q: CodeDistribution) -> float:
    return sum(map(partial(_D, P, Q), P.keys()))

def _D(P: CodeDistribution, Q: CodeDistribution, x: Binary):
    if P[x] == 0:
        return 0
    else:
        if Q[x] == 0:
            return np.inf
        else:
            return P[x]*np.log2(P[x]/Q[x])
    
def memory_entropy(memory: Memory, distribution: CodeDistribution) -> float:
    unique_codes = np.unique(memory[0], axis = 0)
    unique_codes = list(np.apply_along_axis(binary_to_string, 1, unique_codes))
    
    return - sum([shannon_entropy(distribution[code]) for code in unique_codes])


    