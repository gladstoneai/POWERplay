import torch

from .utils import graph
from .utils import misc
from .utils import dist
from .utils import learn
from . import proc

def find_optimal_policy(
    reward_function,
    discount_rate,
    transition_tensor,
    value_initialization=None,
    convergence_threshold=1e-4
):
    return learn.compute_optimal_policy_tensor(
        learn.value_iteration(
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

def run_single_agent_experiment(
    reward_samples,
    mdp_graph,
    discount_rate,
    num_workers=1,
    convergence_threshold=1e-4
):
    print()
    print('Computing POWER samples:')
    print()

    all_optimal_values = proc.samples_to_outputs(
        reward_samples,
        discount_rate,
        graph.graph_to_transition_tensor(mdp_graph),
        iteration_function=learn.value_iteration,
        number_of_samples=len(reward_samples),
        num_workers=num_workers,
        convergence_threshold=convergence_threshold
    )

    power_samples = torch.stack([
        compute_power_values(
            reward_sample, optimal_values, discount_rate
        ) for reward_sample, optimal_values in zip(reward_samples, all_optimal_values)
    ])

    print()
    print('Run complete.')

    return (
        reward_samples,
        power_samples
    )

def run_multiagent_fixed_policy_experiment(
    reward_samples,
    transition_graphs,
    discount_rate,
    num_workers=1,
    convergence_threshold=1e-4
):
    print()
    print('Computing POWER samples:')
    print()

    all_optimal_values = proc.samples_to_outputs(
        reward_samples,
        discount_rate,
        graph.graphs_to_multiagent_transition_tensor(*transition_graphs),
        iteration_function=learn.value_iteration,
        number_of_samples=len(reward_samples),
        num_workers=num_workers,
        convergence_threshold=convergence_threshold
    )

    power_samples = torch.stack([
        compute_power_values(
            reward_sample, optimal_values, discount_rate
        ) for reward_sample, optimal_values in zip(reward_samples, all_optimal_values)
    ])

    print()
    print('Run complete.')

    return (
        reward_samples,
        power_samples
    )

def run_multiagent_with_reward_experiment(
    reward_samples_A,
    reward_samples_B,
    transition_graphs,
    discount_rate,
    num_workers=1,
    convergence_threshold=1e-4
):
    # We precompute these tensors here to avoid recomputation in one of the loops below
    transition_tensor_A = graph.graph_to_transition_tensor(transition_graphs[0])
    transition_tensor_B = graph.graph_to_transition_tensor(transition_graphs[1])

    print()
    print('Computing Agent B policies:')
    print()

    full_transition_tensor_B = graph.graphs_to_multiagent_transition_tensor(
        # NOTE: The order is reversed here since we need (mdp B, policy A, mdp A) to get Agent B's policy
        transition_graphs[1], graph.quick_mdp_to_policy(transition_graphs[0]), transition_graphs[0]
    )

    all_optimal_values_B = proc.samples_to_outputs(
        reward_samples_B,
        discount_rate,
        full_transition_tensor_B,
        iteration_function=learn.value_iteration,
        number_of_samples=len(reward_samples_B),
        num_workers=num_workers,
        convergence_threshold=convergence_threshold
    )

    all_policy_tensors_B = torch.stack([
        learn.compute_optimal_policy_tensor(
            optimal_values, full_transition_tensor_B
        ) for optimal_values in all_optimal_values_B
    ])

    all_full_transition_tensors_A = torch.stack([
        graph.compute_multiagent_transition_tensor(
            transition_tensor_A, policy_tensor_B, transition_tensor_B
        ) for policy_tensor_B in all_policy_tensors_B
    ])

    print()
    print('Computing agent A POWER samples:')
    print()

    all_optimal_values_A = proc.samples_to_outputs(
        reward_samples_A,
        discount_rate,
        all_full_transition_tensors_A,
        iteration_function=learn.value_iteration,
        number_of_samples=len(reward_samples_A),
        num_workers=num_workers,
        convergence_threshold=convergence_threshold
    )

    power_samples_A = torch.stack([
        compute_power_values(
            reward_sample_A, optimal_values_A, discount_rate
        ) for reward_sample_A, optimal_values_A in zip(reward_samples_A, all_optimal_values_A)
    ])

    print()
    print('Run complete.')

    return ( # TODO: Compute POWER proxy for Agent B
        reward_samples_A,
        reward_samples_B,
        power_samples_A
    )

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

    if sweep_type == 'single_agent':
        reward_samples_A, power_samples_A = run_single_agent_experiment(
            reward_samples_agent_A,
            transition_graphs[0],
            discount_rate,
            num_workers=num_workers,
            convergence_threshold=convergence_threshold
        )
        return (
            reward_samples_A,
            None,
            power_samples_A,
            None
        )
    
    elif sweep_type == 'multiagent_fixed_policy':
        reward_samples_A, power_samples_A = run_multiagent_fixed_policy_experiment(
            reward_samples_agent_A,
            transition_graphs,
            discount_rate,
            num_workers=num_workers,
            convergence_threshold=convergence_threshold
        )
        return (
            reward_samples_A,
            None,
            power_samples_A,
            None
        )
    
    elif sweep_type == 'multiagent_with_reward':
        reward_samples_agent_B = dist.generate_correlated_reward_samples(
            reward_sampler, reward_samples_agent_A, correlation=reward_correlation, symmetric_interval=symmetric_interval
        )

        reward_samples_A, reward_samples_B, power_samples_A = run_multiagent_with_reward_experiment(
            reward_samples_agent_A,
            reward_samples_agent_B,
            transition_graphs,
            discount_rate,
            num_workers=num_workers,
            convergence_threshold=convergence_threshold
        )

        return (
            reward_samples_A,
            reward_samples_B,
            power_samples_A,
            None
        )
