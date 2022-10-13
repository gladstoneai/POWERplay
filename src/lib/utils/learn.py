import torch
import copy as cp
import torch.nn.functional as tf

from . import misc

def value_iteration(
    reward_function,
    discount_rate,
    transition_tensor,
    value_initialization=None,
    convergence_threshold=1e-4,
    calculate_as_sparse=False
):
    value_function = torch.zeros(len(reward_function)) if (
        value_initialization is None
    ) else cp.deepcopy(value_initialization)

    transition_tensor_input = misc.sparsify_tensor(transition_tensor) if (
        calculate_as_sparse
    ) else misc.densify_tensor(transition_tensor)

    is_first_iteration = True

    while is_first_iteration or (max_value_change > convergence_threshold):
        is_first_iteration = False
        old_values = cp.deepcopy(value_function)
        
        for state_index in range(len(reward_function)):
            matrix_product = torch.sparse.mm(
                transition_tensor_input[state_index].coalesce(), value_function.unsqueeze(1)
            ) if calculate_as_sparse else torch.matmul(
                transition_tensor_input[state_index], value_function.unsqueeze(1)
            )

            value_function[state_index] = reward_function[state_index] + discount_rate * torch.max(matrix_product)

        max_value_change = misc.calculate_value_convergence(old_values, value_function)

    return value_function

# NOTE: This is from Sutton & Barto, Section 4.1 ("Iterative Policy Evaluation"). 
# As usual, in our setting, the reward r depends only on the current state s, not directly on the action a.
# This means at each iteration, the reward we actually capture always corresponds to the state we are on
# (i.e. reward_function[state]).
def policy_evaluation(
    reward_function,
    discount_rate,
    transition_tensor,
    policy_tensor,
    value_initialization=None,
    convergence_threshold=1e-4,
    calculate_as_sparse=False
):
    value_function_ = torch.zeros(len(reward_function)) if (
        value_initialization is None
    ) else cp.deepcopy(value_initialization)

    transition_tensor_input = misc.sparsify_tensor(transition_tensor) if (
        calculate_as_sparse
    ) else misc.densify_tensor(transition_tensor)

    is_first_iteration = True

    while is_first_iteration or (max_value_change > convergence_threshold):
        is_first_iteration = False
        old_values_ = cp.deepcopy(value_function_)

        for state in range(len(reward_function)):
            matrix_product = torch.sparse.mm(
                transition_tensor_input[state].t().coalesce(), policy_tensor[state].unsqueeze(-1)
            ).t()[0] if calculate_as_sparse else torch.matmul(
                policy_tensor[state], transition_tensor_input[state]
            )

            value_function_[state] = reward_function[state] + discount_rate * torch.dot(matrix_product, value_function_)
            
        max_value_change = misc.calculate_value_convergence(old_values_, value_function_)
    
    return value_function_

# NOTE: in our setting, the reward r depends only on the current state s, not directly on the action a.
# This means we can simplify the argmax expression for the policy (see the last line of the Value Iteration algorithm
# in Section 4.4 of Sutton & Barto) to eliminate the r term and the gamma factor. i.e., we can find the optimal
# policy from the optimal value function by simply taking \argmax_a \sum_{s'} p(s' | s, a) V(s').
# NOTE: whenever two action-values are equal, we intentionally randomize the policy to avoid systematically
# biasing our policies according to the canonical ordering of the states. This matters especially when we use a
# reward function that's sparse (i.e., most states do not yield substantial reward but a few states yield big reward).
# A "tiebreaker" policy between two equally unrewarding states that's very deterministic for e.g. Agent H, is also
# very exploitable for Agent A.
def compute_optimal_policy_tensor(optimal_values, transition_tensor):
    transition_tensor_sparse = misc.sparsify_tensor(transition_tensor)
    action_values = torch.sparse.sum(
        transition_tensor_sparse * optimal_values.expand(
            transition_tensor_sparse.shape
        ).sparse_mask(transition_tensor_sparse.coalesce()),
        dim=2
    ).to_dense()
    return tf.normalize(
        torch.eq(
            action_values,
            action_values.max(dim=1)[0].unsqueeze(1).tile(1, transition_tensor_sparse.shape[1])
        ).to(torch.float32) * torch.sparse.sum(transition_tensor_sparse, dim=2).to_dense(), p=1, dim=1
    )

def find_optimal_policy(
    reward_function,
    discount_rate,
    transition_tensor,
    value_initialization=None,
    convergence_threshold=1e-4
):
    return compute_optimal_policy_tensor(
        value_iteration(
            reward_function,
            discount_rate,
            transition_tensor,
            value_initialization=value_initialization,
            convergence_threshold=convergence_threshold
        ),
        transition_tensor
    )
    
def compute_power_values(reward_sample, optimal_values, discount_rate):
    return ((1 - discount_rate) / discount_rate) * torch.tensor(
        [(
            optimal_values[state_index] - reward_sample[state_index]
        ) for state_index in range(len(optimal_values))]
    )