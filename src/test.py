import torch

from . import launch
from . import data
from . import dist

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
    # 4 states, 3 actions
    # States: 1, 2, 3, 4
    # Actions: left, here, right

    # Rows (axis 0) are states, columns (axis 1) are actions allowed from that state
    state_action_matrix = torch.tensor([
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 0, 0]
    ])

    # Axis 0 elements are starting states, axis 1 elements are actions from that state,
    # axis 2 elements are *ending state* probabilities from that action and that state.
    # If all of axis 2 is zero when axis 0 == i and axis 1 == j, that means action j
    # is forbidden from state i.
    transition_tensor = torch.tensor([
        [[0., 0., 0., 0.], [1., 0., 0., 0.], [0., 1., 0., 0.]],
        [[1., 0., 0., 0.], [0.5, 0.5, 0., 0.], [0., 0.2, 0.8, 0.]],
        [[0., 1., 0., 0.], [0., 0.3, 0.7, 0.], [0., 0., 0.4, 0.6]],
        [[0., 0., 0.9, 0.1], [0., 0., 0., 0.], [0., 0., 0., 0.]]
    ])

    state_list = ['1', '2', '3', '4']
    discount_rate = 0.8

    reward_sampler = dist.config_to_reward_distribution(
        state_list, {
            'default_dist': {
                'dist_name': 'uniform',
                'params': [0, 1]
            }
        }
    )

    return launch.run_one_experiment(
        state_action_matrix,
        discount_rate,
        reward_sampler,
        transition_tensor=transition_tensor,
        num_reward_samples=30000,
        num_workers=10
    )