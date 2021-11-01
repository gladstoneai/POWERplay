import numpy as np
import copy as cp

NEG_INFTY = -1e309
TINY_NUMBER = 1e-256 # To avoid both divide by 0 and overflow

def check_adjacency_matrix(adjacency_matrix):
    if (not (adjacency_matrix[-1][:-1] == 0).all()) or (not adjacency_matrix[-1][-1] == 1):
        raise Exception(
            'The last row of the adjacency matrix must be 1 in the last entry, 0 elsewhere. '\
            'The last entry represents the terminal state, which can only lead to itself.'
        )
    
    return cp.deepcopy(adjacency_matrix)

def check_policy(policy):
    if np.shape(policy)[0] != np.shape(policy)[1]:
        raise Exception('The policy array must be n x n.')
    
    if (np.sum(policy, axis=1) != 1).any():
        raise Exception('Each row of the policy array must sum to 1.')
    
    return cp.deepcopy(policy)

def check_value_reward(value_or_reward):
    if value_or_reward[-1] != 0:
        raise Exception(
            'The last entry of a value or reward function must be 0,'\
            'because this corresponds to the terminal state.'
        )
    
    return cp.deepcopy(value_or_reward)

def mask_adjacency_matrix(adjacency_matrix, neg_infty=NEG_INFTY):
    masked_adjacency_matrix = np.ma.masked_where(
        adjacency_matrix == 0, adjacency_matrix.astype(float), copy=True
    )
# Whenever a fill value is multiplied by anything, the original fill value is returned at that entry.
    masked_adjacency_matrix.set_fill_value(neg_infty)

    return masked_adjacency_matrix

# Biggest fractional change in state value over one iteration.
def calculate_value_convergence(old_values, new_values, tiny_number=TINY_NUMBER):
    return np.max(np.abs((old_values - new_values) / (old_values + tiny_number)))

def generate_random_policy(number_of_states, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    raw_policy = np.random.rand(number_of_states, number_of_states)
# Each row of the policy sums to 1, since it represents the probabilities of actions from that state.
    return raw_policy / np.sum(raw_policy, axis=1, keepdims=True)

# Random rewards over states, sample uniformly over interval. The last state listed is conventionally
# the terminal state, and always has reward 0. (The same must hold for all value initializations.)
def generate_random_reward(number_of_states, interval=(0, 1), seed=None):
    if interval[0] >= interval[1]:
        raise Exception('The interval must be a 2-tuple whose second entry is bigger than its first.')

    if seed is not None:
        np.random.seed(seed)

    return np.append((np.random.rand(number_of_states - 1) * (interval[1] - interval[0])) + interval[0], 0)