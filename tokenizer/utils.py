'''
Utility functions for the development of a tokenizer
'''

def get_stats(ids):
    '''
    Given an input list of ids, this function will return a dictionary
    with the counts of each pair of ids.

    Args:
    ids: list of integers

    Returns:
    counts: dictionary of pairs of integers and their counts
    '''
    counts = {}
    for pair in zip(ids, ids[1:]):
        if pair not in counts:
            counts[pair] = 0
        counts[pair] += 1
    return counts


def merge(ids, pair, idx):
    '''
    Given an input list of ids, a pair of ids, and an index, this function
    will merge the pair of ids into a new token with the given index.

    Args:
    ids: list of integers
    pair: tuple of integers
    idx: integer

    Returns:
    new_ids: list of integers
    '''
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids