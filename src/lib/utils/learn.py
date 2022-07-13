import torch
import copy as cp

from . import misc

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

# NOTE: This is from Sutton & Barto, Section 4.1 ("Iterative Policy Evaluation").
def policy_evaluation(
    reward_function,
    discount_rate,
    transition_tensor,
    policy_tensor,
    value_initialization=None,
    convergence_threshold=1e-4
):
    value_function_ = torch.zeros(len(reward_function)) if (
        value_initialization is None
    ) else cp.deepcopy(value_initialization)

    is_first_iteration = True

    while is_first_iteration or (max_value_change > convergence_threshold):
        is_first_iteration = False

        max_value_change = 0
        old_values_ = cp.deepcopy(value_function_)

        for state in range(len(reward_function)):
            value_function_[state] = torch.sum(
                torch.matmul(
                    policy_tensor[state], transition_tensor[state]
                ) * (
                    reward_function[state] + discount_rate * value_function_[state]
                )
            )
            
        max_value_change = misc.calculate_value_convergence(old_values_, value_function_)
    
    return value_function_

# NOTE: in our setting, the reward r depends only on the current state s, not directly on the action a.
# This means we can simplify the argmax expression for the policy (see the last line of the Value Iteration algorithm
# in Section 4.4 of Sutton & Barto) to eliminate the r term and the gamma factor. i.e., we can find the optimal
# policy from the optimal value function by simply taking \argmax_a \sum_{s'} p(s' | s, a) V(s').
def compute_optimal_policy_tensor(optimal_values, transition_tensor):
    return torch.sparse_coo_tensor(
        torch.stack((
            torch.arange(transition_tensor.shape[0]),
            torch.argmax(torch.matmul(transition_tensor, optimal_values.unsqueeze(1)), dim=1).flatten()
        )),
        torch.ones(transition_tensor.shape[0]),
        size=tuple(transition_tensor.shape[:2])
    ).to_dense()
