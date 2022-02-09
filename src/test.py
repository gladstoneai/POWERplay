from . import launch
from . import data

def test_vanilla():
    launch.launch_sweep(
        'test_sweep.yaml',
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        project='uncategorized'
    )

def test_gridworld():
    launch.launch_sweep(
        'test_sweep_gridworld.yaml',
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        project='uncategorized',
        plot_as_gridworld=True
    )

def test_stochastic():
    launch.launch_sweep(
        'test_sweep_stochastic.yaml',
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        project='uncategorized'
    )
