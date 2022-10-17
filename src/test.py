import pathlib as path

from .lib.utils import misc
from .lib import data
from . import launch

def test_vanilla():
    launch.launch_sweep(
        'test_sweep.yaml',
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        plot_distributions=True,
        plot_correlations=True,
        project='uncategorized',
        sweep_config_folder=path.Path()/data.SWEEP_CONFIGS_FOLDER/'test',
        force_basic_font=True
    )

def test_gridworld():
    launch.launch_sweep(
        'test_sweep_gridworld.yaml',
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        project='uncategorized',
        plot_distributions=True,
        plot_correlations=True,
        plot_as_gridworld=True,
        sweep_config_folder=path.Path()/data.SWEEP_CONFIGS_FOLDER/'test'
    )

def test_stochastic():
    launch.launch_sweep(
        'test_sweep_stochastic.yaml',
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        plot_distributions=True,
        plot_correlations=True,
        project='uncategorized',
        sweep_config_folder=path.Path()/data.SWEEP_CONFIGS_FOLDER/'test'
    )

def test_multiagent():
    launch.launch_sweep(
        'test_run_multi_simulated.yaml',
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        sweep_local_id=misc.generate_sweep_id(),
        plot_distributions=True,
        plot_correlations=True,
        project='uncategorized',
        sweep_config_folder=path.Path()/data.SWEEP_CONFIGS_FOLDER/'test'
    )

    launch.launch_sweep(
        'test_run_multi_actual.yaml',
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        sweep_local_id=misc.generate_sweep_id(),
        project='uncategorized',
        plot_distributions=True,
        plot_correlations=True,
        sweep_config_folder=path.Path()/data.SWEEP_CONFIGS_FOLDER/'test'
    )

def test_reward_correlation():
    launch.launch_sweep(
        'test_sweep_correlated_rewards.yaml',
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        project='uncategorized',
        plot_as_gridworld=True,
        plot_distributions=True,
        plot_correlations=False,
        diagnostic_mode=True,
        sweep_config_folder=path.Path()/data.SWEEP_CONFIGS_FOLDER/'test'
    )