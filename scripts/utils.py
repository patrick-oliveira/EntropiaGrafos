from typing import List
from itertools import islice

def split_list(input_list: List, number_of_slices: int) -> List[List]:
    '''
    Create a list of slices from 'input_list' of fixed size.
    
    If L = len(input_list) and n = number_of_slices, then the Division Algorithm says that there are integers q and r < n such that L = q*n + r. If L is divisible by n, then the output list will be a set of n slices of size q, otherwise, there will be r slices of size q + 1 and and n - r slices of size q.
    
    Args:
        input_list (List): List to be segmented into sublists of potentially equal size.
        segment_size (Int): Number of slices.
        
    Output:
        output_segments (List[List]): A list of slices of fixed size 
    '''
    L = len(input_list)
    n = number_of_slices
    input_list = iter(input_list)
    
    if L// n != 0:
        split_groups_sizes = [L//n]*n
        for i in range(L%n):
            split_groups_sizes[i] = split_groups_sizes[i] + 1
    else:  
        split_groups_sizes = [L]
    
    output_segments = [list(islice(input_list, size)) for size in split_groups_sizes]
    
    return output_segments if len(output_segments) != 1 else output_segments[0]