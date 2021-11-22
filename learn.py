import torch.nn.functional as tf
import torch
import copy as cp
import sys

import utils

def get_greedy_policy(value_function, adjacency_matrix):
    return tf.one_hot(torch.argmax(
        adjacency_matrix * (value_function + torch.min(value_function) + 1), axis=1
    ), num_classes=len(value_function))

def value_iteration(
    reward_function,
    discount_rate,
    adjacency_matrix,
    value_initialization=None,
    convergence_threshold=1e-4
):
    value_function = torch.zeros(len(reward_function)) if (
        value_initialization is None
    ) else cp.deepcopy(value_initialization)

    is_first_iteration = True

    while is_first_iteration or (max_value_change > convergence_threshold):
        is_first_iteration = False

        max_value_change = 0
        old_values = cp.deepcopy(value_function)
        
        for state in range(len(reward_function)):
            value_function[state] = reward_function[state] + discount_rate * torch.max(
                value_function.masked_select(adjacency_matrix[state].type(torch.ByteTensor).bool()
                )
            )

        max_value_change = utils.calculate_value_convergence(old_values, value_function)

    return (
        get_greedy_policy(value_function, adjacency_matrix),
        value_function
    )

def power_calculation_constructor(
    adjacency_matrix,
    discount_rate,
    convergence_threshold=1e-4,
    value_initializations=None,
    worker_pool_size=1
):
    def power_sample_calculator(reward_samples, worker_id=1):
        all_value_initializations = [None] * len(reward_samples) if (
            value_initializations is None
        ) else value_initializations

        all_power_samples = []

        for i in range(len(reward_samples)):
            if worker_id == 0: # Only the first worker prints so the pool isn't slowed
                sys.stdout.write('Running samples {0} / {1}'.format(
                    worker_pool_size * (i + 1), worker_pool_size * len(reward_samples)
                ))
                sys.stdout.flush()
                sys.stdout.write('\r')
                sys.stdout.flush()

            optimal_values = value_iteration(
                reward_samples[i],
                discount_rate,
                adjacency_matrix,
                value_initialization=all_value_initializations[i],
                convergence_threshold=convergence_threshold,
            )[1]

            # NOTE (IMPORTANT): tensor multiplication between optimal_values and reward_samples[i] CANNOT be
            # used here without causing multiproessing to hang when the TOTAL number of reward samples
            # is >=40k or so for 10 workers. The source of this bug is unknown but may be related to an
            # idiosyncracy of pathos multiprocessing. Even just CREATING a reward tensor of that size in the
            # main process and having ANY operation on reward_samples in the child process - EVEN if the
            # reward_samples tensor is NOT RELATED to the created 40k+ tensor outside - is sufficient to
            # reproduce this bug. There is obviously some kind of side effect at play here. Explicitly
            # making all operations on reward_samples element-wise patches the bug, because these ops don't
            # "count" as being done on the reward_samples tensor, but only on its elements.
            all_power_samples += [((1 - discount_rate) / discount_rate) * torch.tensor(
                [(optimal_values[state] - reward_samples[i][state]) for state in range(len(optimal_values))]
            )]

        if worker_id == 0:
            print() # Jump to newline after stdout.flush()

        return torch.stack(all_power_samples)
    
    return power_sample_calculator

def calculate_power_samples(*args, **kwargs):
    return power_calculation_constructor(*args, **kwargs)(0)