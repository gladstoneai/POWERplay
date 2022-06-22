import torch
import copy as cp
import sys
import functools as func
import multiprocessing as mps

from src.lib.utils import graph
from .utils import misc
from .utils import dist
from . import check

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

# Note: in our setting, the reward r depends only on the current state s, not directly on the action a.
# This means we can simplify the argmax expression for the policy (see the last line of the Value Iteration algorithm
# in Section 4.4 of Sutton & Barto) to eliminate the r term and the gamma factor. i.e., we can find the optimal
# policy from the optimal value function by simply taking \argmax_a \sum_{s'} p(s' | s, a) V(s').
def compute_optimal_policy_tensor(_, optimal_values, __, transition_tensor):
    return torch.sparse_coo_tensor(
        torch.stack((
            torch.arange(transition_tensor.shape[0]),
            torch.argmax(torch.matmul(transition_tensor, optimal_values.unsqueeze(1)), dim=1).flatten()
        )),
        torch.ones(transition_tensor.shape[0]),
        size=tuple(transition_tensor.shape[:2])
    ).to_dense()

def evaluate_optimal_policy(
    reward_function,
    discount_rate,
    transition_tensor,
    value_initialization=None,
    convergence_threshold=1e-4
):
    return compute_optimal_policy_tensor(
        reward_function,
        value_iteration(
            reward_function,
            discount_rate,
            transition_tensor,
            value_initialization=value_initialization,
            convergence_threshold=convergence_threshold
        ),
        discount_rate,
        transition_tensor
    )

def compute_power_values(reward_sample, optimal_values, discount_rate, _):
    return ((1 - discount_rate) / discount_rate) * torch.tensor(
        [(optimal_values[state] - reward_sample[state]) for state in range(len(optimal_values))]
    )

def output_sample_calculator(
    reward_samples,
    transition_tensors,
    discount_rate,
    compute_output_quantity=compute_power_values,
    convergence_threshold=1e-4,
    value_initializations=None,
    worker_pool_size=1,
    worker_id=0
):

    all_value_initializations = [None] * len(reward_samples) if (
        value_initializations is None
    ) else value_initializations

    all_output_samples_ = []

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
            transition_tensors[i],
            value_initialization=all_value_initializations[i],
            convergence_threshold=convergence_threshold
        )

        all_output_samples_ += [
            compute_output_quantity(reward_samples[i], optimal_values, discount_rate, transition_tensors[i])
        ]

    if worker_id == 0:
        print() # Jump to newline after stdout.flush()

    return torch.stack(all_output_samples_)

def output_sample_calculator_mps(discount_rate, reward_samples, transition_tensors, worker_id, **kwargs):
    return output_sample_calculator(reward_samples, transition_tensors, discount_rate, **{
        **kwargs,
        **{ 'worker_id': worker_id }
    })

def rewards_to_outputs(
    reward_samples,
    all_transition_tensors,
    discount_rate,
    compute_output_quantity=compute_power_values,
    num_workers=1,
    convergence_threshold=1e-4
):
    check.check_num_samples(len(reward_samples), num_workers)
    check.check_num_samples(len(all_transition_tensors), num_workers)

    output_calculator = func.partial(
        output_sample_calculator_mps,
        discount_rate,
        compute_output_quantity=compute_output_quantity,
        convergence_threshold=convergence_threshold,
        value_initializations=None,
        worker_pool_size=num_workers
    )

    with mps.Pool(num_workers) as pool:
        output_samples_list = pool.starmap(
            output_calculator,
            zip(
                torch.split(reward_samples, len(reward_samples) // num_workers, dim=0),
                torch.split(all_transition_tensors, len(all_transition_tensors) // num_workers, dim=0),
                range(num_workers)
            )
        )
    
    return torch.cat(output_samples_list, axis=0)

def run_one_experiment(
    transition_graphs,
    discount_rate,
    reward_sampler,
    sweep_type='single_agent',
    num_reward_samples=10000,
    num_workers=1,
    convergence_threshold=1e-4,
    random_seed=None,
    reward_correlation=None,
    reward_noise=None
):
    misc.set_global_random_seed(random_seed)
    
    # When the number of samples doesn't divide evenly into the available workers, truncate the samples
    reward_samples_agent_A = reward_sampler(num_workers * (num_reward_samples // num_workers))

    if sweep_type == 'single_agent':

        mdp_graph_A = transition_graphs[0]
        all_transition_tensors = torch.tile(
            graph.graph_to_transition_tensor(mdp_graph_A),
            (len(reward_samples_agent_A), 1, 1, 1)
        )
    
    elif sweep_type == 'multiagent_fixed_policy':

        mdp_graph_A, policy_graph_B, mdp_graph_B = transition_graphs
        all_transition_tensors = torch.tile(
            graph.graphs_to_multiagent_transition_tensor(mdp_graph_A, policy_graph_B, mdp_graph_B),
            (len(reward_samples_agent_A), 1, 1, 1)
        )
    
    elif sweep_type == 'multiagent_with_reward':

        mdp_graph_A, mdp_graph_B = transition_graphs
        reward_samples_agent_B = dist.generate_correlated_reward_samples(
            reward_sampler, reward_samples_agent_A, correlation=reward_correlation, noise=reward_noise
        )

        print('Computing Agent B policies:')
        print()
        '''
        policies_agent_B = rewards_to_outputs(
            reward_samples_agent_B,
            base_transition_tensors, # TODO: Compute the multiagent transition tensor from mdp_graph_A, mdp_graph_B, plus the agent A uniform random policy
            discount_rate,
            compute_output_quantity=compute_optimal_policy_tensor,
            num_workers=num_workers,
            convergence_threshold=convergence_threshold
        )
        '''
        # TODO: Add code to combine the Agent B policy tensor with the base_transition_tensor to get the full transition tensor list

    print('Computing POWER samples:')
    print()

    power_samples = rewards_to_outputs(
        reward_samples_agent_A,
        all_transition_tensors,
        discount_rate,
        compute_output_quantity=compute_power_values,
        num_workers=num_workers,
        convergence_threshold=convergence_threshold
    )

    print()
    print('Run complete.')

    return (
        reward_samples_agent_A, # TODO: Return Agent B reward samples and Agent B policies
        power_samples
    )
