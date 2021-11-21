import pathos.multiprocessing as mps
import torch
import pathlib as path
import time
import wandb as wb
import subprocess as sp
import re
import os
import json

import utils
import data
import wb_api
import learn
import dist
import check
import viz

def run_one_experiment_OLD(
    adjacency_matrix=None,
    state_list=None,
    discount_rate=None,
    reward_distribution=None,
    num_reward_samples=1000,
    convergence_threshold=1e-4,
    num_workers=1,
    random_seed=None,
    save_experiment_local=True,
    save_experiment_wandb=True,
    wandb_run_params={},
    experiment_handle='',
    experiment_id=time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())),
    save_folder=data.EXPERIMENT_FOLDER,
    plot_when_done=False,
    save_figs=False
):
    check.check_experiment_inputs(
        adjacency_matrix=adjacency_matrix,
        discount_rate=discount_rate,
        reward_distribution=reward_distribution,
        state_list=state_list,
        plot_when_done=plot_when_done,
        save_figs=save_figs,
        num_workers=num_workers,
        save_experiment_wandb=save_experiment_wandb,
        wandb_run_params=wandb_run_params
    )

    experiment_name = '{0}-{1}'.format(experiment_id, experiment_handle)

    if random_seed is None:
        torch.seed()
    else:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

    samples_per_worker = num_reward_samples // num_workers
    num_reward_samples_actual = num_workers * samples_per_worker

    reward_samples = dist.reward_distribution_constructor(state_list)(num_reward_samples_actual) if (
        reward_distribution is None
    ) else reward_distribution(num_reward_samples_actual)

    experiment_config = {
        'adjacency_matrix': adjacency_matrix,
        'state_list': state_list,
        'discount_rate': discount_rate,
        'num_reward_samples': num_reward_samples,
        'num_reward_samples_actual': num_reward_samples_actual,
        'convergence_threshold': convergence_threshold,
        'reward_distribution': reward_distribution,
        'num_workers': num_workers,
        'random_seed': random_seed
    }

    if save_experiment_wandb:
        wb_api.login()

    wb_tracker = wb_api.initialize_run(
        experiment_name, wandb_run_params, experiment_config
    ) if save_experiment_wandb else None

    st_power_calculator = learn.power_calculation_constructor(
        adjacency_matrix,
        discount_rate,
        convergence_threshold=convergence_threshold,
        value_initializations=None,
        worker_pool_size=num_workers
    )

    with mps.ProcessingPool(num_workers) as pool:
        power_samples_list = pool.map(
            st_power_calculator, torch.split(reward_samples, samples_per_worker, dim=0), range(num_workers)
        )

    power_samples = torch.cat(power_samples_list, axis=0)

    rendered_outputs = viz.render_all_outputs(
        reward_samples,
        power_samples,
        state_list,
        show=plot_when_done,
        save_handle=experiment_name,
        save_figure=data.save_figure if save_figs else None,
        save_folder=path.Path(save_folder)/experiment_name
    )

    if wb_tracker is not None:
        wb_tracker.log({ fig_name: wb.Image(fig) for fig_name, fig in rendered_outputs.items() })
    
    experiment = {
        'name': experiment_name,
        'inputs': experiment_config,
        'outputs': {
            'reward_samples': reward_samples,
            'power_samples': power_samples
        }
    }

    if save_experiment_local:
        data.save_experiment(experiment, folder=path.Path(save_folder)/experiment_name)
    
    if wb_tracker is not None:
        wb_tracker.finish()
    
    print()
    print('Run complete.')
    print()
    print(experiment_name)

    return experiment

def rewards_to_powers(
    reward_samples,
    adjacency_matrix,
    discount_rate,
    num_workers=1,
    convergence_threshold=1e-4
):
    check.check_num_samples(len(reward_samples), num_workers)

    st_power_calculator = learn.power_calculation_constructor(
        adjacency_matrix,
        discount_rate,
        convergence_threshold=convergence_threshold,
        value_initializations=None,
        worker_pool_size=num_workers
    )

    with mps.ProcessingPool(num_workers) as pool:
        power_samples_list = pool.map(
            st_power_calculator,
            torch.split(reward_samples, len(reward_samples) // num_workers, dim=0),
            range(num_workers)
        )

    return torch.cat(power_samples_list, axis=0)

def run_one_experiment(
    adjacency_matrix,
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
        adjacency_matrix,
        discount_rate,
        num_workers=num_workers,
        convergence_threshold=convergence_threshold
    )

    return (
        reward_samples,
        power_samples
    )

def launch_sweep(
    sweep_config_filename,
    sweep_config_folder='.',
    sweep_local_id=time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())),
    entity=data.get_settings_value(data.WANDB_ENTITY_PATH, settings_filename=data.SETTINGS_FILENAME),
    project='uncategorized',
    output_folder_local=data.EXPERIMENT_FOLDER
):
    input_sweep_config = data.load_sweep_config(sweep_config_filename, folder=path.Path(sweep_config_folder))
    check.check_sweep_params(input_sweep_config.get('parameters'))

    sweep_name = '{0}-{1}'.format(sweep_local_id, input_sweep_config.get('name'))

    output_sweep_config = utils.build_sweep_config(
        input_sweep_config=input_sweep_config,
        sweep_name=sweep_name,
        project=project,
        entity=entity
    )
    sweep_config_filepath = data.save_sweep_config(
        output_sweep_config, folder=path.Path(output_folder_local)/sweep_name
    )

    sp.run(['wandb', 'login'])
    sp.run([
        'wandb',
        'agent',
        re.search(
            r'(?<=wandb agent )(.*)',
            sp.run(['wandb', 'sweep', sweep_config_filepath], capture_output=True).stderr.decode('utf-8')
        ).group()
    ], env={
        'LOCAL_SWEEP_NAME': sweep_name,
        'SWEEP_VARIABLE_PARAMS': json.dumps([
            param for param in input_sweep_config.get('parameters').keys() if (
                input_sweep_config.get('parameters').get(param).get('values') is not None
            )
        ]),
        **os.environ
    })


