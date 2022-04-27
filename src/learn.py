import torch
import copy as cp
import sys

from .utils import misc

def value_iteration(
    reward_function,
    discount_rate,
    transition_tensor,
    value_initialization=None,
    convergence_threshold=1e-4
):
    value_function = torch.zeros(len(reward_function)) if (
        value_initialization is None
    ) else cp.deepcopy(value_initialization)
    transition_tensor_sparse = transition_tensor.to_sparse()

    is_first_iteration = True

    while is_first_iteration or (max_value_change > convergence_threshold):
        is_first_iteration = False

        max_value_change = 0
        old_values = cp.deepcopy(value_function)
        
        for state in range(len(reward_function)):
            value_function[state] = reward_function[state] + discount_rate * torch.max(
                torch.sparse.mm(transition_tensor_sparse[state], value_function.unsqueeze(1))
            )

        max_value_change = misc.calculate_value_convergence(old_values, value_function)

    return value_function

def power_sample_calculator(
    reward_samples,
    transition_tensor,
    discount_rate,
    convergence_threshold=1e-4,
    value_initializations=None,
    worker_pool_size=1,
    worker_id=0
):

    all_value_initializations = [None] * len(reward_samples) if (
        value_initializations is None
    ) else value_initializations

    all_power_samples_ = []

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
            transition_tensor,
            value_initialization=all_value_initializations[i],
            convergence_threshold=convergence_threshold
        )

        all_power_samples_ += [((1 - discount_rate) / discount_rate) * torch.tensor(
            [(optimal_values[state] - reward_samples[i][state]) for state in range(len(optimal_values))]
        )]

    if worker_id == 0:
        print() # Jump to newline after stdout.flush()

    return torch.stack(all_power_samples_)

def power_sample_calculator_mps(state_action_matrix, discount_rate, reward_samples, worker_id, **kwargs):
    return power_sample_calculator(reward_samples, state_action_matrix, discount_rate, **{
        **kwargs,
        **{ 'worker_id': worker_id }
    })

# Note: in our setting, the reward r depends only on the current state s, not directly on the action a.
# This means we can simplify the argmax expression for the policy (see the last line of the Value Iteration algorithm
# in Section 4.4 of Sutton & Barto) to eliminate the r term and the gamma factor. i.e., we can find the optimal
# policy from the optimal value function by simply taking \argmax_a \sum_{s'} p(s' | s, a) V(s').
def compute_optimal_policy_tensor(
    reward_function,
    discount_rate,
    transition_tensor,
    value_initialization=None,
    convergence_threshold=1e-4
):
    return torch.sparse_coo_tensor(
        torch.stack((
            torch.arange(transition_tensor.shape[0]),
            torch.argmax(
                torch.matmul(
                    transition_tensor,
                    value_iteration(
                        reward_function,
                        discount_rate,
                        transition_tensor,
                        value_initialization=value_initialization,
                        convergence_threshold=convergence_threshold
                    ).unsqueeze(1)
                ), dim=1
            ).flatten()
        )),
        torch.ones(transition_tensor.shape[0]),
        size=tuple(transition_tensor.shape[:2])
    ).to_dense()