import numpy as np
from Scripts.Types import CodeDistribution
from Scripts.Parameters import code_length


def shannon_entropy(P: float) -> float:
    return P*np.log2(P)

def memory_entropy(distribution: CodeDistribution) -> float:
    P = distribution[distribution > 0]
    return - shannon_entropy(P).sum()
            
# def JSD(Pu: CodeDistribution, Pv: CodeDistribution) -> float:
#     M = (Pu + Pv)/2
#     return (D(Pu, M) + D(Pv, M))/2

def JSD(Pu: CodeDistribution, Pv: CodeDistribution) -> float:
    M = (Pu + Pv)/2
    return memory_entropy(M) - (memory_entropy(Pu) + memory_entropy(Pv))/2

def D(P: CodeDistribution, Q: CodeDistribution) -> float:
    '''
    The quotient P/Q will never be infinity because the only divergence being computed is between the
    probabilities P, Q and M, where M = (P + Q)/2

    Parameters
    ----------
    P : CodeDistribution
        DESCRIPTION.
    Q : CodeDistribution
        DESCRIPTION.

    Returns
    -------
    float
        DESCRIPTION.

    '''
    not_zero_indexes = np.arange(2**code_length)[(P > 0) * (Q > 0)]
    Q_not_zero = Q[not_zero_indexes]
    P_not_zero = P[not_zero_indexes]
    
    return (P_not_zero*np.log2(P_not_zero/Q_not_zero)).sum()

# def _D(P: CodeDistribution, Q: CodeDistribution, x: Binary):
#     if P[x] == 0:
#         return 0
#     else:
#         if Q[x] == 0:
#             return np.inf
#         else:
#             return P[x]*np.log2(P[x]/Q[x])
