import numpy as np
from scripts.Types import CodeDistribution, Binary, Memory
from scripts.Parameters import code_length
from scripts.Memory import binary_to_string
from functools import partial


def shannon_entropy(P: float) -> float:
    return P*np.log2(P)


def memory_entropy(distribution: CodeDistribution) -> float:
    P = np.asarray(list(distribution.values()))
    P = P[P > 0]
    return - shannon_entropy(P).sum()
            
def JSD(Pu: CodeDistribution, Pv: CodeDistribution) -> float:
    M = {code:(Pu[code] + Pv[code])/2 for code in Pu}
    
    return memory_entropy(M) - (memory_entropy(Pu) + memory_entropy(Pv))/2

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

def S(Pu: CodeDistribution, Pv: CodeDistribution) -> float:
    return 1 - JSD(Pu, Pv)