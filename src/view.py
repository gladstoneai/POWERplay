import matplotlib.pyplot as plt
import torch
import numpy as np

from .lib.utils import render
from .lib.utils import graph
from .lib.utils import dist
from .lib.utils import misc
from .lib import data
from .lib import get
from . import anim
from . import viz

def plot_gridworld_rollout(
    state_list,
    state_rollout,
    reward_function=None,
    show=True,
    ms_per_frame=200,
    agent_whose_rewards_are_displayed='H',
    save_handle='gridworld_rollout',
    save_folder=data.TEMP_FOLDER
):
    all_figs = [
        render.render_gridworld_rollout_snapshot(
            state_list,
            current_state,
            reward_function=reward_function,
            agent_whose_rewards_are_displayed=agent_whose_rewards_are_displayed
        ) for current_state in state_rollout
    ]
    fig_names = ['{0}-{1}'.format(save_handle, i) for i in range(len(state_rollout))]

    for fig, fig_name in zip(all_figs, fig_names):
        data.save_figure(fig, fig_name, folder=save_folder)
    
    anim.animate_from_filenames(
        fig_names,
        'rollout_animation',
        ms_per_frame=ms_per_frame,
        input_folder_or_list=save_folder,
        output_folder=save_folder
    )

    if show:
        plt.show()

    return all_figs

def plot_policy_sample(
    policy_graph,
    reward_function,
    discount_rate,
    show=True,
    subgraphs_per_row=4,
    number_of_states_per_figure=128,
    save_handle='temp',
    save_folder=data.TEMP_FOLDER,
    temp_folder=data.TEMP_FOLDER
):
    return viz.plot_mdp_or_policy(
        policy_graph,
        show=show,
        subgraphs_per_row=subgraphs_per_row,
        number_of_states_per_figure=number_of_states_per_figure,
        reward_to_plot=reward_function,
        discount_rate_to_plot=discount_rate,
        save_handle=save_handle,
        graph_name='policy_reward_sample',
        save_folder=save_folder,
        temp_folder=temp_folder
    )

# y_axis_bounds is either None (and set the y-axis bounds by default based on the data) or a 2-element list
# with the lower bound as the first element, and the upper bound as the second element.
def plot_alignment_curves(
    sweep_id,
    show=True,
    plot_title='Alignment curves',
    fig_name=None,
    y_axis_bounds=None,
    include_baseline_power=True,
    data_folder=data.EXPERIMENT_FOLDER,
    save_folder=data.TEMP_FOLDER
):
    power_correlation_data = get.get_reward_correlations_and_powers_from_sweep(
        sweep_id, include_baseline_power=include_baseline_power, folder=data_folder
    )

    reward_correlations, all_avg_powers_H, all_avg_powers_A, baseline_avg_power_H = (
        power_correlation_data['reward_correlations'],
        [powers_H.mean() for powers_H in power_correlation_data['all_powers_H']],
        [powers_A.mean() for powers_A in power_correlation_data['all_powers_A']],
        power_correlation_data.get('baseline_powers_H', torch.tensor(0.)).mean()
    )

    fig, ax = plt.subplots()

    ax.plot(reward_correlations, all_avg_powers_A, 'rs', label='Agent A POWER, avg over states')
    ax.plot(reward_correlations, all_avg_powers_H, 'bo', label='Agent H POWER, avg over states')
    
    if include_baseline_power:
        ax.plot(
            reward_correlations,
            [baseline_avg_power_H] * len(reward_correlations),
            'b-',
            label='Agent H baseline POWER, avg over states'
        )

    ax.set_xlabel('Reward correlation coefficient')
    ax.set_ylabel('POWER averages')
    ax.set_title(plot_title)

    if y_axis_bounds is not None:
        ax.set_ylim(y_axis_bounds)

    ax.legend()

    if fig_name is not None:
        data.save_figure(
            fig,
            '{0}-sweep_id_{1}-alignment_curve'.format(fig_name, sweep_id),
            folder=save_folder
        )

    if show:
        plt.show()

# all_powers_H and all_powers_A are lists of tensors.
def plot_power_correlation_relationship(
    x_axis_values,
    all_powers_H,
    all_powers_A,
    show=True,
    fig_name='temp',
    x_axis_label='Reward correlation value',
    include_zero_line=True,
    save_folder=data.TEMP_FOLDER
):
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)

    power_correlations = [
        torch.corrcoef(torch.stack([powers_H, powers_A]))[0][1] for (
            powers_H, powers_A
        ) in zip(all_powers_H, all_powers_A)
    ]

    if include_zero_line:
        ax.plot(x_axis_values, [0] * len(x_axis_values), 'm-', alpha=1)

    ax.plot(x_axis_values, power_correlations, 'mo', markeredgewidth=0, alpha=0.75)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel('State-by-state POWER correlation value')
    ax.set_title('Correlation coefficients plot')

    data.save_figure(
        fig,
        '{}-power_correlation_relationship'.format(fig_name),
        folder=save_folder
    )

    if show:
        plt.show()

def plot_specific_power_alignments(
    sweep_id,
    show=True,
    fig_name='temp',
    include_baseline_powers=True,
    graph_padding_x=0.05,
    graph_padding_y=0.05,
    ms_per_frame=200,
    frames_at_start=1,
    frames_at_end=1,
    include_zero_line=True,
    data_folder=data.EXPERIMENT_FOLDER,
    save_folder=data.TEMP_FOLDER
):
    power_correlation_data = get.get_reward_correlations_and_powers_from_sweep(
        sweep_id, include_baseline_power=include_baseline_powers, folder=data_folder
    )

    reward_correlations, all_powers_H, all_powers_A, baseline_powers_H = (
        power_correlation_data['reward_correlations'],
        power_correlation_data['all_powers_H'],
        power_correlation_data['all_powers_A'],
        power_correlation_data.get('baseline_powers_H')
    )

    min_H_power = min([powers_H.min() for powers_H in all_powers_H]) * (1 - graph_padding_x)
    max_H_power = max([powers_H.max() for powers_H in all_powers_H]) * (1 + graph_padding_x)
    min_A_power = min([powers_A.min() for powers_A in all_powers_A]) * (1 - graph_padding_y)
    max_A_power = max([powers_A.max() for powers_A in all_powers_A]) * (1 + graph_padding_y)

    power_correlations_ = []
    all_fig_names_ = []

    for correlation, powers_H, powers_A in zip(reward_correlations, all_powers_H, all_powers_A):
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 6)

        current_fig_name = '{0}-sweep_id_{1}-specific_alignment_curve-correlation_{2}'.format(
            fig_name, sweep_id, str(correlation)
        )

        if include_baseline_powers:
            for baseline_power in baseline_powers_H:
                ax.plot([baseline_power.item()] * 2, [min_A_power, max_A_power], 'b-', alpha=0.25)

        ax.plot(powers_H, powers_A, 'mo', markeredgewidth=0, alpha=0.25)

        ax.set_xlabel('State POWER values, Agent H')
        ax.set_ylabel('State POWER values, Agent A')
        ax.set_title('State POWER values for Agents H and A, reward correlation = {:.2f}'.format(correlation))
        ax.set_xlim([min_H_power, max_H_power])
        ax.set_ylim([min_A_power, max_A_power])

        data.save_figure(fig, current_fig_name, folder=save_folder)

        power_correlations_ += [torch.corrcoef(torch.stack([powers_H, powers_A]))[0][1]]
        all_fig_names_ += [current_fig_name]

    anim.animate_from_filenames(
        all_fig_names_,
        '{}-animation'.format(fig_name),
        ms_per_frame=ms_per_frame,
        frames_at_start=frames_at_start,
        frames_at_end=frames_at_end,
        input_folder_or_list=save_folder,
        output_folder=save_folder
    )

    plot_power_correlation_relationship(
        reward_correlations,
        all_powers_H,
        all_powers_A,
        x_axis_label='Reward correlation value',
        include_zero_line=include_zero_line,
        show=show,
        fig_name='{0}-sweep_id_{1}-reward_correlation-'.format(fig_name, sweep_id),
        save_folder=save_folder
    )

    if show:
        plt.show()

def plot_correlated_reward_samples(
    num_samples,
    distribution_config={ 'dist_name': 'uniform', 'params': [0, 1] },
    correlation=0,
    symmetric_interval=None,
    random_seed=0,
    xlim=None,
    ylim=None,
    show=True,
    fig_name=None,
    distribution_dict=dist.DISTRIBUTION_DICT,
    save_folder=data.TEMP_FOLDER
):
    if random_seed:
        misc.set_global_random_seed(random_seed)

    agent_H_dist = dist.config_to_reward_distribution(
        ['TEMP'], { 'default_dist': distribution_config }, distribution_dict=distribution_dict
    )
    agent_H_samples = agent_H_dist(num_samples)
    agent_A_samples = dist.generate_correlated_reward_samples(
        agent_H_dist, agent_H_samples, correlation=correlation, symmetric_interval=symmetric_interval
    )

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)

    ax.plot(
        np.array(agent_H_samples.T[0]), np.array(agent_A_samples.T[0]), 'go', markeredgewidth=0, alpha=0.25
    )
    ax.set_xlabel('Agent H reward samples')
    ax.set_ylabel('Agent A reward samples')
    ax.set_title('Agent H and A reward values, correlation = {:.2f}'.format(correlation))

    if xlim is not None:
        ax.set_xlim(xlim)
    
    if ylim is not None:
        ax.set_ylim(ylim)

    if fig_name is not None:
        data.save_figure(fig, fig_name, folder=save_folder)

    if show:
        plt.show()

def view_gridworld(gridworld_mdp):
    viz.plot_sample_aggregations(
        torch.zeros((1, len(graph.get_states_from_graph(gridworld_mdp)))),
        graph.get_states_from_graph(gridworld_mdp),
        plot_as_gridworld=True,
        sample_quantity='',
        sample_units='dummy data'
    )