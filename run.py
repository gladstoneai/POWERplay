import pathos.multiprocessing as mps
import torch
import pathlib as path
import time
import wandb as wb

import utils
import data
import viz
import wb_api
import learn
import dist
import check

def run_one_experiment(
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

    reward_sampler = dist.reward_distribution_constructor(state_list) if (
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
        num_reward_samples=samples_per_worker,
        reward_distribution=reward_sampler,
        convergence_threshold=convergence_threshold,
        value_initializations=None,
        random_seed=random_seed,
        worker_pool_size=num_workers
    )

    with mps.ProcessingPool(num_workers) as pool:
        all_samples_list = pool.map(st_power_calculator, range(num_workers))

    reward_samples = torch.cat([samples_group[0] for samples_group in all_samples_list], axis=0)
    power_samples = torch.cat([samples_group[1] for samples_group in all_samples_list], axis=0)

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

def run_one_experiment_TEMP(config):
    return config

def run_experiment_sweep(
    sweep_params,
    sweep_name='test_sweep',
    sweep_description='',
    project='test-project',
    entity=data.get_settings_value(data.WANDB_ENTITY_PATH, settings_filename=data.SETTINGS_FILENAME),
    sweep_id=time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())),
    save_outputs_local=False,
    save_folder_local=data.EXPERIMENT_FOLDER,
    plot_when_done=True
):
    check.check_sweep_params(sweep_params)
    sweep_config = utils.build_sweep_config(
        sweep_name=sweep_name,
        sweep_id=sweep_id,
        sweep_params=sweep_params,
        sweep_description=sweep_description,
        project=project,
        entity=entity
    )
    sweep_folder = path.Path()/save_folder_local/sweep_config.get('name')

    wb_api.login() # TODO: Refactor out all instances of wb_api in favor of directly calling wb

    if save_outputs_local:
        data.save_experiment({ 'name': sweep_config.get('name'), 'inputs': sweep_params }, folder=sweep_folder)

    def run_one():
        with wb.init() as run:
            run_params = {
                key: (value if ('value' in sweep_params.get(key)) else value[0]) for key, value in run.config.items()
            }
            # Hack to force a specified name for the run in W&B interface
            run.name = '-'.join([sweep_config.get('name')] + [
                '{0}__{1}'.format(key, value[1]) for key, value in run.config.items() if (
                    'value' not in sweep_params.get(key)
                )
             ])

            # TODO: Add reward function and adjacency matrix interpreter, since they now come in as primitives
            print('HERE') ##
            print(run.config) ##
            print(run_params) ##

            reward_samples = run.config.get('adjacency_matrix')[0]
            power_samples = run.config.get('adjacency_matrix')[1]

            if save_outputs_local:
                data.save_experiment({
                    'name': run.name,
                    'inputs': run_params,
                    'outputs': {
                        'reward_samples': reward_samples, 'power_samples': power_samples
                    }
                }, folder=sweep_folder/run.name)
                # TODO NEXT: Add figure rendering & saving, etc.
    wb.agent(wb.sweep(sweep_config), function=run_one)

# TODO: Write a sweep wrapper that splits inputs into an experiment config (for reproducibility) and
# a set of measurement params (like save_figs, etc.). Should take in a config as input.
# This should use the W&B sweep API (https://docs.wandb.ai/guides/sweeps/python-api).
# TODO: Document API for the sweep function.
# TODO: Add ability to run sweep without wandb server access (i.e., offline mode). May be impossible, but
# would be great as it would allow me to run local tests without consuming bandwidth, etc.
# TODO: Test run of this sweep function.
# TODO: Transition from old experiment function to new experiment function.
# TODO: Delete load_experiment function.

