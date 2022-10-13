import torch

from .utils import graph
from .utils import misc
from .utils import dist
from .utils import learn
from . import proc

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
        graph.graph_to_full_transition_tensor(mdp_graph, return_sparse=True),
        iteration_function=learn.value_iteration,
        number_of_samples=len(reward_samples),
        num_workers=num_workers,
        convergence_threshold=convergence_threshold
    )

    power_samples = torch.stack([
        learn.compute_power_values(
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
        graph.graphs_to_full_transition_tensor(
            *transition_graphs, acting_agent_is_H=True, return_sparse=True
        ),
        iteration_function=learn.value_iteration,
        number_of_samples=len(reward_samples),
        num_workers=num_workers,
        convergence_threshold=convergence_threshold
    )

    power_samples = torch.stack([
        learn.compute_power_values(
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
    reward_samples_H,
    reward_samples_A,
    transition_graphs,
    discount_rate_agent_H,
    discount_rate_agent_A,
    num_workers=1,
    convergence_threshold=1e-4,
    diagnostic_mode=False
):
    # We precompute these tensors here to avoid recomputation in one of the loops below
    joint_transition_tensor = graph.graph_to_joint_transition_tensor(transition_graphs[0], return_sparse=True)
    seed_policy_tensor_A = graph.graph_to_policy_tensor(transition_graphs[1], return_sparse=False) # The fixed policy tensor we give Agent A, that Agent H initially optimizes against

    print()
    print('Computing Agent H policies:')
    print()

    full_transition_tensor_H = graph.compute_full_transition_tensor( # Full Agent H transition tensor assuming Agent A seed policy
        joint_transition_tensor, seed_policy_tensor_A, acting_agent_is_H=True, return_sparse=True
    )

    all_optimal_values_H = proc.samples_to_outputs(
        reward_samples_H,
        discount_rate_agent_H,
        full_transition_tensor_H,
        iteration_function=learn.value_iteration,
        number_of_samples=len(reward_samples_H),
        num_workers=num_workers,
        convergence_threshold=convergence_threshold
    )

    all_policy_tensors_H = torch.stack([
        learn.compute_optimal_policy_tensor(
            optimal_values_H, full_transition_tensor_H
        ) for optimal_values_H in all_optimal_values_H
    ])

    all_full_transition_tensors_A = torch.stack([
        graph.compute_full_transition_tensor(
            joint_transition_tensor, policy_tensor_H, acting_agent_is_H=False, return_sparse=True
        ) for policy_tensor_H in all_policy_tensors_H
    ])
    
    if diagnostic_mode:
        print()
        print('Computing Agent A POWER samples (seed policy):')
        print()

        power_samples_H_against_seed = torch.stack([
            learn.compute_power_values(
                reward_sample_H, optimal_values_H, discount_rate_agent_H
            ) for reward_sample_H, optimal_values_H in zip(reward_samples_H, all_optimal_values_H)
        ])

        all_values_A = proc.samples_to_outputs(
            reward_samples_A,
            discount_rate_agent_A,
            all_full_transition_tensors_A,
            seed_policy_tensor_A,
            iteration_function=learn.policy_evaluation,
            number_of_samples=len(reward_samples_A),
            num_workers=num_workers,
            convergence_threshold=convergence_threshold
        )

        power_samples_A_seed = torch.stack([
            learn.compute_power_values(
                reward_sample_A, values_A, discount_rate_agent_A
            ) for reward_sample_A, values_A in zip(reward_samples_A, all_values_A)
        ])

    print()
    print('Computing Agent A POWER samples:')
    print()

    all_optimal_values_A = proc.samples_to_outputs(
        reward_samples_A,
        discount_rate_agent_A,
        all_full_transition_tensors_A,
        iteration_function=learn.value_iteration,
        number_of_samples=len(reward_samples_A),
        num_workers=num_workers,
        convergence_threshold=convergence_threshold
    )

    power_samples_A = torch.stack([
        learn.compute_power_values(
            reward_sample_A, optimal_values_A, discount_rate_agent_A
        ) for reward_sample_A, optimal_values_A in zip(reward_samples_A, all_optimal_values_A)
    ])

    print()
    print('Computing Agent H POWER samples:')
    print()

    all_policy_tensors_A = torch.stack([
        learn.compute_optimal_policy_tensor(
            optimal_values_A, full_transition_tensor_A_chunk[0]
        ) for optimal_values_A, full_transition_tensor_A_chunk in zip(
            all_optimal_values_A, misc.chunk_1d_tensor_into_list(all_full_transition_tensors_A, len(all_optimal_values_A))
        )
    ])

    all_full_transition_tensors_H = torch.stack([
        graph.compute_full_transition_tensor(
            joint_transition_tensor, policy_tensor_A, acting_agent_is_H=True, return_sparse=True
        ) for policy_tensor_A in all_policy_tensors_A
    ])
    
    all_values_H = proc.samples_to_outputs(
        reward_samples_H,
        discount_rate_agent_H,
        all_full_transition_tensors_H,
        all_policy_tensors_H,
        iteration_function=learn.policy_evaluation,
        number_of_samples=len(reward_samples_H),
        num_workers=num_workers,
        convergence_threshold=convergence_threshold
    )

    power_samples_H = torch.stack([
        learn.compute_power_values(
            reward_sample_H, values_H, discount_rate_agent_H
        ) for reward_sample_H, values_H in zip(reward_samples_H, all_values_H)
    ])

    print()
    print('Run complete.')

    return (
        reward_samples_H,
        reward_samples_A,
        power_samples_H,
        power_samples_A,
        {} if not diagnostic_mode else {
            'all_values_H_against_seed': all_optimal_values_H,
            'all_values_H_against_A': all_values_H,
            'all_values_A_seed_policy': all_values_A,
            'power_samples_H_against_seed': power_samples_H_against_seed,
            'power_samples_A_seed_policy': power_samples_A_seed
        }
    )

def run_one_experiment(
    transition_graphs,
    discount_rate_agent_H,
    reward_sampler,
    sweep_type='single_agent',
    num_reward_samples=10000,
    num_workers=1,
    convergence_threshold=1e-4,
    random_seed=None,
    reward_correlation=None,
    symmetric_interval=None,
    discount_rate_agent_A=None,
    diagnostic_mode=False
):
    misc.set_global_random_seed(random_seed)
    # When the number of samples doesn't divide evenly into the available workers, truncate the samples
    reward_samples_agent_H = reward_sampler(num_workers * (num_reward_samples // num_workers))

    if sweep_type == 'single_agent':
        reward_samples_H, power_samples_H = run_single_agent_experiment(
            reward_samples_agent_H,
            transition_graphs[0],
            discount_rate_agent_H,
            num_workers=num_workers,
            convergence_threshold=convergence_threshold
        )

        return (
            reward_samples_H,
            None,
            power_samples_H,
            None,
            {}
        )
    
    elif sweep_type == 'multiagent_fixed_policy':
        reward_samples_H, power_samples_H = run_multiagent_fixed_policy_experiment(
            reward_samples_agent_H,
            transition_graphs,
            discount_rate_agent_H,
            num_workers=num_workers,
            convergence_threshold=convergence_threshold
        )

        return (
            reward_samples_H,
            None,
            power_samples_H,
            None,
            {}
        )
    
    elif sweep_type == 'multiagent_with_reward':
        reward_samples_agent_A = dist.generate_correlated_reward_samples(
            reward_sampler,
            reward_samples_agent_H,
            correlation=reward_correlation,
            symmetric_interval=symmetric_interval
        )

        (
            reward_samples_H, reward_samples_A, power_samples_H, power_samples_A, diagnostic_dict
        ) = run_multiagent_with_reward_experiment(
            reward_samples_agent_H,
            reward_samples_agent_A,
            transition_graphs,
            discount_rate_agent_H,
            discount_rate_agent_A,
            num_workers=num_workers,
            convergence_threshold=convergence_threshold,
            diagnostic_mode=diagnostic_mode
        )

        return (
            reward_samples_H,
            reward_samples_A,
            power_samples_H,
            power_samples_A,
            diagnostic_dict
        )
