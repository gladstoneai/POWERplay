import pathos.multiprocessing as mps
import torch
import pathlib as path
import time
import subprocess as sp
import re
import os
import json

import utils
import data
import learn
import check

def rewards_to_powers(
    reward_samples,
    adjacency_matrix,
    discount_rate,
    num_workers=1,
    convergence_threshold=1e-4,
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

    print()
    print('Run complete.')

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
    output_folder_local=data.EXPERIMENT_FOLDER,
    beep_when_done=False
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

# NOTE: The wandb Python API uses multithreading, which makes it incompatible with matplotlib.pyplot rendering.
# The CLI does not use multithreading, and is compatible with pyplot, so we run the CLI via Python subprocess.
# (This behavior by wandb is undocumented.)
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

    if beep_when_done:
        print('\a')
        print('\a')
        print('\a')



