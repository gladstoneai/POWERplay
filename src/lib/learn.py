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
        [(optimal_values[state] - reward_sample[state]) for state in range(len(optimal_values))]
    )

def output_sample_calculator(
    *args,
    number_of_samples=1,
    iteration_function=value_iteration,
    convergence_threshold=1e-4,
    value_initializations=None,
    worker_pool_size=1,
    worker_id=0
):
    args_by_iteration = [[args[i][j] for i in range(len(args))] for j in range(number_of_samples)]
    all_value_initializations = [None] * number_of_samples if (
        value_initializations is None
    ) else value_initializations

    all_output_samples_ = []

    for i in range(number_of_samples):
        if worker_id == 0: # Only the first worker prints so the pool isn't slowed
            sys.stdout.write('Running samples {0} / {1}'.format(
                worker_pool_size * (i + 1), worker_pool_size * number_of_samples
            ))
            sys.stdout.flush()
            sys.stdout.write('\r')
            sys.stdout.flush()

        all_output_samples_ += [
            iteration_function(
                *args_by_iteration[i],
                value_initialization=all_value_initializations[i],
                convergence_threshold=convergence_threshold
            )
        ]

    if worker_id == 0:
        print() # Jump to newline after stdout.flush()

    return torch.stack(all_output_samples_)

def output_sample_calculator_mps(worker_id, *args, **kwargs):
    return output_sample_calculator(*args, **{ **kwargs, **{ 'worker_id': worker_id } })

def samples_to_outputs(
    *args,
    number_of_samples=1,
    iteration_function=value_iteration,
    num_workers=1,
    convergence_threshold=1e-4
):
    check.check_num_samples(number_of_samples, num_workers)
    check.check_tensor_or_number_args(args)

    output_calculator = func.partial(
        output_sample_calculator_mps,
        number_of_samples=number_of_samples // num_workers,
        iteration_function=iteration_function,
        convergence_threshold=convergence_threshold,
        value_initializations=None,
        worker_pool_size=num_workers
    )

    with mps.Pool(num_workers) as pool:
        output_samples_list = pool.starmap(
            output_calculator,
            zip(
                range(num_workers),
                *[torch.split(
                    tiled_arg, number_of_samples // num_workers, dim=0
                ) for tiled_arg in [
                    arg if (
                        hasattr(arg, '__len__') and len(arg) == number_of_samples
                    ) else misc.tile_tensor(arg, number_of_samples) for arg in args
                ]]
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
    symmetric_interval=None
):
    misc.set_global_random_seed(random_seed)
    
    # When the number of samples doesn't divide evenly into the available workers, truncate the samples
    reward_samples_agent_A = reward_sampler(num_workers * (num_reward_samples // num_workers))
    reward_samples_agent_B_ = None

    if sweep_type == 'single_agent':

        transition_tensor_A = graph.graph_to_transition_tensor(transition_graphs[0])
        full_transition_tensors_A = misc.tile_tensor(transition_tensor_A, len(reward_samples_agent_A))
    
    elif sweep_type == 'multiagent_fixed_policy':

        transition_tensor_A, policy_tensor_B, transition_tensor_B = (
            graph.graph_to_transition_tensor(transition_graphs[0]),
            graph.graph_to_policy_tensor(transition_graphs[1]),
            graph.graph_to_transition_tensor(transition_graphs[2])
        )
        full_transition_tensors_A = misc.tile_tensor(
            graph.compute_multiagent_transition_tensor(transition_tensor_A, policy_tensor_B, transition_tensor_B),
            len(reward_samples_agent_A)
        )
    
    elif sweep_type == 'multiagent_with_reward':

        policy_tensor_A_random, transition_tensor_A, transition_tensor_B = (
            graph.graph_to_policy_tensor(graph.quick_mdp_to_policy(transition_graphs[0])),
            graph.graph_to_transition_tensor(transition_graphs[0]),
            graph.graph_to_transition_tensor(transition_graphs[1])
        )
        reward_samples_agent_B_ = dist.generate_correlated_reward_samples(
            reward_sampler, reward_samples_agent_A, correlation=reward_correlation, symmetric_interval=symmetric_interval
        )
        
        print()
        print('Computing Agent B policies:')
        print()

        full_transition_tensor_B = graph.compute_multiagent_transition_tensor(
            transition_tensor_B, policy_tensor_A_random, transition_tensor_A # NOTE: The order is reversed here since we need (mdp B, policy A, mdp A) to get Agent B's policy
        )

        policy_tensors_B = torch.stack(
            [compute_optimal_policy_tensor(
                optimal_values, full_transition_tensor_B
            ) for optimal_values in samples_to_outputs(
                    reward_samples_agent_B_,
                    discount_rate,
                    full_transition_tensor_B,
                    iteration_function=value_iteration,
                    number_of_samples=len(reward_samples_agent_B_),
                    num_workers=num_workers,
                    convergence_threshold=convergence_threshold
                )
            ]
        )
        
        full_transition_tensors_A = torch.stack([graph.compute_multiagent_transition_tensor(
            transition_tensor_A, policy_tens_B, transition_tensor_B
        ) for policy_tens_B in policy_tensors_B])

    print()
    print('Computing agent A POWER samples:')
    print()

    power_samples_agent_A = torch.stack(
        [compute_power_values(
            reward_sample, optimal_values, discount_rate
        ) for reward_sample, optimal_values in zip(
            reward_samples_agent_A,
            samples_to_outputs(
                reward_samples_agent_A,
                discount_rate,
                full_transition_tensors_A,
                iteration_function=value_iteration,
                number_of_samples=len(reward_samples_agent_A),
                num_workers=num_workers,
                convergence_threshold=convergence_threshold
            )
        )])
    
    if sweep_type == 'multiagent_with_reward':
        print()
        print('Computing agent B POWER samples:')
        print()
    
    power_samples_agent_B = None # TODO: Delete and use if statement above instead

    print()
    print('Run complete.')

    return (
        reward_samples_agent_A,
        reward_samples_agent_B_,
        power_samples_agent_A,
        power_samples_agent_B
    )
