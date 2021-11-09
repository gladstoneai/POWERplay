import copy as cp
import sys
import uuid
import pathos.multiprocessing as mps
import torch
import torch.nn.functional as tf

import utils
import data
import viz
import wb_api

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

        power_samples = ((1 - discount_rate) / discount_rate) * (
            torch.stack(all_optimal_values) - torch.stack(all_reward_functions)
        )

        if worker_id == 0:
            print() # Jump to newline after stdout.flush()

        return power_samples
    
    return power_sample_calculator

def calculate_power_samples(*args, **kwargs):
    return power_calculation_constructor(*args, **kwargs)(0)

def run_one_experiment(
    adjacency_matrix=None,
    state_list=None,
    discount_rate=None,
    reward_distribution=None,
    num_reward_samples=1000,
    convergence_threshold=1e-4,
    value_initializations=None,
    num_workers=1,
    random_seed=None,
    save_experiment_local=True,
    save_experiment_wandb=True,
    wandb_run_params={},
    experiment_handle='',
    save_folder=data.EXPERIMENT_FOLDER,
    plot_when_done=False,
    save_figs=False
):
    utils.check_experiment_inputs(
        adjacency_matrix=adjacency_matrix,
        discount_rate=discount_rate,
        num_reward_samples=num_reward_samples,
        reward_distribution=reward_distribution,
        value_initializations=value_initializations,
        state_list=state_list,
        plot_when_done=plot_when_done,
        save_figs=save_figs,
        num_workers=num_workers,
        save_experiment_wandb=save_experiment_wandb,
        wandb_run_params=wandb_run_params
    )

    experiment_name = '{0}-{1}'.format(experiment_handle, uuid.uuid4().hex)

    reward_sampler = utils.reward_distribution_constructor(state_list) if (
        reward_distribution is None
    ) else reward_distribution
    samples_per_worker = num_reward_samples // num_workers
    num_reward_samples_actual = num_workers * samples_per_worker

    experiment_config = {
        'adjacency_matrix': adjacency_matrix,
        'state_list': state_list,
        'discount_rate': discount_rate,
        'num_reward_samples': num_reward_samples,
        'num_reward_samples_actual': num_reward_samples_actual,
        'reward_distribution': reward_sampler,
        'convergence_threshold': convergence_threshold,
        'value_initializations': value_initializations,
        'num_workers': num_workers,
        'random_seed': random_seed
    }

    if save_experiment_wandb:
        wb_api.login()
    
    wb_tracker = wb_api.initialize_run(
        experiment_name, wandb_run_params, experiment_config
    ) if save_experiment_wandb else None

    st_power_calculator = power_calculation_constructor(
        adjacency_matrix,
        discount_rate,
        num_reward_samples=samples_per_worker,
        reward_distribution=reward_sampler,
        convergence_threshold=convergence_threshold,
        value_initializations=value_initializations,
        random_seed=random_seed,
        worker_pool_size=num_workers
    )

    with mps.ProcessingPool(num_workers) as pool:
        power_samples_list = pool.map(st_power_calculator, range(num_workers))

    power_samples = torch.cat(power_samples_list, axis=0)
    reward_samples = reward_sampler(num_reward_samples_actual)

    print()
    print(experiment_name)

    viz_kwargs = {
        'show': plot_when_done,
        'save_fig': save_figs,
        'save_handle': experiment_name,
        'save_folder': save_folder,
        'wb_tracker': wb_tracker
    }

    viz.plot_power_means(power_samples, state_list, **viz_kwargs)
    viz.plot_reward_samples(reward_samples, state_list, **viz_kwargs)
    viz.plot_power_samples(power_samples, state_list, **viz_kwargs)
    
    for state in state_list[:-1]: # Don't plot or save terminal state
        viz.plot_power_correlations(power_samples, state_list, state, **viz_kwargs)
    
    experiment = {
        'name': experiment_name,
        'inputs': experiment_config,
        'outputs': {
            'reward_samples': reward_samples,
            'power_samples': power_samples
        }
    }

    if save_experiment_local:
        data.save_experiment(experiment, folder=save_folder)
    
    if wb_tracker is not None:
        wb_tracker.finish()
    
    return experiment
    