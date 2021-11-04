import numpy as np
import copy as cp
import sys
import uuid
import pathos.multiprocessing as mps

import utils
import data
import viz

def get_greedy_policy(value_function, adjacency_matrix):
    policy_actions = np.argmax(utils.mask_adjacency_matrix(adjacency_matrix) * value_function, axis=1)
    policy = np.zeros((len(value_function), len(value_function)))
    policy[np.arange(len(policy)), policy_actions] = 1

    return policy

def value_iteration(
    reward_function,
    discount_rate,
    adjacency_matrix,
    value_initialization=None,
    convergence_threshold=1e-4
):
    value_function = np.zeros(len(reward_function)) if (
        value_initialization is None
    ) else cp.deepcopy(value_initialization)

    is_first_iteration = True

    while is_first_iteration or (max_value_change > convergence_threshold):
        is_first_iteration = False

        max_value_change = 0
        old_values = cp.deepcopy(value_function)
        
        for state in range(len(reward_function)):
            value_function[state] = reward_function[state] + discount_rate * np.max(
                utils.mask_adjacency_matrix(adjacency_matrix[state]) * value_function
            )

        max_value_change = utils.calculate_value_convergence(old_values, value_function)
    
    return (
        get_greedy_policy(value_function, adjacency_matrix),
        value_function
    )

def power_calculation_factory(
    adjacency_matrix,
    discount_rate,
    num_reward_samples=1000,
    reward_distributions=lambda x: 1,
    reward_ranges=(0, 1),
    reward_sample_resolution=100,
    convergence_threshold=1e-4,
    value_initializations=None,
    random_seed=None,
    worker_pool_size=1
):

    def power_sample_calculator(worker_id):
        if random_seed is None:
            np.random.seed()
        else:
            np.random.seed(random_seed + worker_id) # Force a different seed for each worker
        
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

            all_reward_functions += [
                utils.generate_random_reward(
                    len(adjacency_matrix),
                    target_distributions=reward_distributions,
                    intervals=reward_ranges,
                    resolution=reward_sample_resolution,
                    seed=None
                )
            ]
            all_optimal_values += [
                value_iteration(
                    all_reward_functions[-1],
                    discount_rate,
                    adjacency_matrix,
                    value_initialization=all_value_initializations[i],
                    convergence_threshold=convergence_threshold,
                )[1]
            ]
        
        if worker_id == 0:
            print() # Jump to newline after stdout.flush()

        power_samples = ((1 - discount_rate) / discount_rate) * (
            np.stack(all_optimal_values) - np.stack(all_reward_functions)
        )

        return power_samples
    
    return power_sample_calculator

def calculate_power_samples(*args, **kwargs):
    return power_calculation_factory(*args, **kwargs)(0)

def run_one_experiment(
    adjacency_matrix=None,
    discount_rate=None,
    num_reward_samples=1000,
    reward_distributions=lambda x: 1,
    reward_ranges=(0, 1),
    reward_sample_resolution=100,
    convergence_threshold=1e-4,
    value_initializations=None,
    num_workers=1,
    random_seed=None,
    save_experiment=True,
    experiment_handle='',
    save_folder=data.EXPERIMENT_FOLDER,
    plot_when_done=False,
    save_figs=False,
    state_list=None
):
    utils.check_experiment_inputs(
        adjacency_matrix=adjacency_matrix,
        discount_rate=discount_rate,
        num_reward_samples=num_reward_samples,
        reward_distributions=reward_distributions,
        reward_ranges=reward_ranges,
        value_initializations=value_initializations,
        reward_sample_resolution=reward_sample_resolution,
        state_list=state_list,
        plot_when_done=plot_when_done,
        save_figs=save_figs,
        num_workers=num_workers
    )

    samples_per_worker = num_reward_samples // num_workers

    st_power_calculator = power_calculation_factory(
        adjacency_matrix,
        discount_rate,
        num_reward_samples=samples_per_worker,
        reward_distributions=reward_distributions,
        reward_ranges=reward_ranges,
        reward_sample_resolution=reward_sample_resolution,
        convergence_threshold=convergence_threshold,
        value_initializations=value_initializations,
        random_seed=random_seed,
        worker_pool_size=num_workers
    )

    with mps.ProcessingPool(num_workers) as pool:
        power_samples_list = pool.map(st_power_calculator, range(num_workers))

    power_samples = np.concatenate(power_samples_list, axis=0)

    experiment = {
        'name': '{0}-{1}'.format(experiment_handle, uuid.uuid4().hex),
        'inputs': {
            'adjacency_matrix': adjacency_matrix,
            'discount_rate': discount_rate,
            'num_reward_samples': num_reward_samples,
            'num_reward_samples_actual': num_workers * samples_per_worker,
            'reward_distributions': reward_distributions,
            'reward_ranges': reward_ranges,
            'reward_sample_resolution': reward_sample_resolution,
            'convergence_threshold': convergence_threshold,
            'value_initializations': value_initializations,
            'random_seed': random_seed
        },
        'outputs': {
            'powers': np.mean(power_samples, axis=0),
            'power_samples': power_samples
        }
    }

    print()
    print(experiment['name'])

    if save_experiment:
        data.save_experiment(experiment, folder=save_folder)

    if state_list is not None:
        kwargs = {
            'show': plot_when_done,
            'save_fig': save_figs,
            'save_handle': experiment['name'],
            'save_folder': save_folder
        }

        viz.plot_power_means(experiment['outputs']['power_samples'], state_list, **kwargs)
        viz.plot_power_samples(experiment['outputs']['power_samples'], state_list, **kwargs)

        for state in state_list[:-1]: # Don't plot or save terminal state
            viz.plot_power_correlations(
                experiment['outputs']['power_samples'], state_list, state, **kwargs
            )
    
    return experiment
    