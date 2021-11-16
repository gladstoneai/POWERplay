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
    num_reward_samples=1000,
    reward_distribution=None,
    convergence_threshold=1e-4,
    value_initializations=None,
    random_seed=None,
    worker_pool_size=1
):
    def power_sample_calculator(worker_id):
        if random_seed is None:
            torch.seed()
        else:
            torch.manual_seed(random_seed + worker_id) # Force a different seed for each worker
            torch.cuda.manual_seed(random_seed + worker_id)
        
        all_value_initializations = [None] * num_reward_samples if (
            value_initializations is None
        ) else value_initializations

        all_optimal_values = []
        all_reward_functions = []

        for i in range(num_reward_samples):
            if worker_id == 0: # Only the first worker prints so the pool isn't slowed
                sys.stdout.write('Running samples {0} / {1}'.format(
                    worker_pool_size * (i + 1), worker_pool_size * num_reward_samples
                ))
                sys.stdout.flush()
                sys.stdout.write('\r')
                sys.stdout.flush()

            all_reward_functions += [reward_distribution(1)[0]]
            all_optimal_values += [
                value_iteration(
                    all_reward_functions[-1],
                    discount_rate,
                    adjacency_matrix,
                    value_initialization=all_value_initializations[i],
                    convergence_threshold=convergence_threshold
                )[1]
            ]

        reward_samples = torch.stack(all_reward_functions)
        power_samples = ((1 - discount_rate) / discount_rate) * (
            torch.stack(all_optimal_values) - reward_samples
        )

        if worker_id == 0:
            print() # Jump to newline after stdout.flush()

        return (
            reward_samples,
            power_samples
        )
    
    return power_sample_calculator

def calculate_power_samples(*args, **kwargs):
    return power_calculation_constructor(*args, **kwargs)(0)