import multiprocessing as mps
import torch
import functools as func

from .utils import graph
from . import data
from . import learn
from . import check

def compute_transition_tensor(
    run_params,
    mdps_folder=data.MDPS_FOLDER,
    policies_folder=data.POLICIES_FOLDER
):
    if run_params.get('mdp_graph') is not None: # Single agent case
        mdp_graph = data.load_graph_from_dot_file(run_params.get('mdp_graph'), folder=mdps_folder)
        transition_tensor = graph.graph_to_transition_tensor(mdp_graph)
        graphs_output = [mdp_graph]

    else: # Multiagent case
        mdp_graph_A = data.load_graph_from_dot_file(
            run_params.get('mdp_graph_agent_A'), folder=mdps_folder
        )
        mdp_graph_B = data.load_graph_from_dot_file(
            run_params.get('mdp_graph_agent_B'), folder=mdps_folder
        )
        policy_graph_B = data.load_graph_from_dot_file(
            run_params.get('policy_graph_agent_B'), folder=policies_folder
        )

        transition_tensor = graph.graphs_to_multiagent_transition_tensor(
            mdp_graph_A, mdp_graph_B, policy_graph_B
        )
        graphs_output = [mdp_graph_A, mdp_graph_B, policy_graph_B]

    return [
        transition_tensor,
        graphs_output
    ]

def rewards_to_powers(
    reward_samples,
    transition_tensor,
    discount_rate,
    num_workers=1,
    convergence_threshold=1e-4
):
    check.check_num_samples(len(reward_samples), num_workers)

    power_calculator = func.partial(
        learn.power_sample_calculator_mps,
        transition_tensor,
        discount_rate,
        convergence_threshold=convergence_threshold,
        value_initializations=None,
        worker_pool_size=num_workers
    )

    with mps.Pool(num_workers) as pool:
        power_samples_list = pool.starmap(
            power_calculator,
            zip(
                torch.split(reward_samples, len(reward_samples) // num_workers, dim=0),
                range(num_workers)
            )
        )
    
    return torch.cat(power_samples_list, axis=0)

def run_one_experiment(
    transition_tensor,
    discount_rate,
    reward_sampler,
    num_reward_samples=10000,
    num_workers=1,
    convergence_threshold=1e-4,
    random_seed=None
):
    if random_seed is None:
        torch.seed()
    else:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
    
    # When the number of samples doesn't divide evenly into the available workers, truncate the samples
    reward_samples = reward_sampler(num_workers * (num_reward_samples // num_workers))

    power_samples = rewards_to_powers(
        reward_samples,
        transition_tensor,
        discount_rate,
        num_workers=num_workers,
        convergence_threshold=convergence_threshold
    )

    print()
    print('Run complete.')

    return (
        reward_samples,
        power_samples
    )
