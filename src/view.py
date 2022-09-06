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
from . import policy

def plot_gridworld_rollout(
    state_list,
    state_rollout,
    reward_function=None,
    show=True,
    ms_per_frame=200,
    agent_whose_rewards_are_displayed='A',
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

# TODO: Document.
def plot_policy_sample(
    policy_graph,
    reward_function,
    discount_rate,
    show=True,
    subgraphs_per_row=4,
    number_of_states_per_figure=128,
    save_handle='temp',
    save_folder=data.TEMP_FOLDER,
    temp_folder=data.TEMP_FOLDER,
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

# NOTE: This function is not optimized for speed and runs very slowly, even for just 10 rollouts.
# It's also not capable of saving the plot and just shows it.
# run_properties: outupt of get.get_properties_from_run(sweep_id, run_suffix=run_suffix)
def plot_rollout_powers(run_properties, initial_state, number_of_rollouts=10, number_of_steps=20):
    state_list = graph.get_states_from_graph(run_properties['transition_graphs'][0])
    powers_A = run_properties['power_samples'].mean(dim=0)

    all_power_plot_rollouts_ = []

    for reward_sample_index in range(number_of_rollouts):

        print('Building rollout for reward sample {}...'.format(reward_sample_index))

        policy_data = policy.sample_optimal_policy_data_from_run(
            run_properties, reward_sample_index=reward_sample_index
        )

        rollout = policy.simulate_policy_rollout(
            initial_state,
            policy_data['policy_graph_A'],
            policy_data['mdp_graph'],
            policy_graph_B=policy_data['policy_graph_B'],
            number_of_steps=number_of_steps
        )

        power_rollout = [powers_A[state_list.index(state)].item() for state in rollout]

        plt.plot(list(range(number_of_steps + 1)), power_rollout, 'b-')
        all_power_plot_rollouts_ += [power_rollout]

    plt.show()

    return all_power_plot_rollouts_

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

    reward_correlations, all_avg_powers_A, all_avg_powers_B, baseline_avg_power_A = (
        power_correlation_data['reward_correlations'],
        [powers_A.mean() for powers_A in power_correlation_data['all_powers_A']],
        [powers_B.mean() for powers_B in power_correlation_data['all_powers_B']],
        power_correlation_data.get('baseline_powers_A', torch.tensor(0.)).mean()
    )

    fig, ax = plt.subplots()

    ax.plot(reward_correlations, all_avg_powers_B, 'rs', label='Agent B POWER, avg over states')
    ax.plot(reward_correlations, all_avg_powers_A, 'bo', label='Agent A POWER, avg over states')
    
    if include_baseline_power:
        ax.plot(
            reward_correlations,
            [baseline_avg_power_A] * len(reward_correlations),
            'b-',
            label='Agent A baseline POWER, avg over states'
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

# reward_correlations, all_powers_A, and all_powers_B are lists of tensors.
def plot_power_correlation_relationship(
    x_axis_values,
    all_powers_A,
    all_powers_B,
    show=True,
    fig_name='temp',
    x_axis_label='Reward correlation value',
    save_folder=data.TEMP_FOLDER
):
    fig, ax = plt.subplots()

    power_correlations = [
        torch.corrcoef(torch.stack([powers_A, powers_B]))[0][1] for (
            powers_A, powers_B
        ) in zip(all_powers_A, all_powers_B)
    ]

    ax.plot(x_axis_values, power_correlations, 'mh')
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
    ms_per_frame=200,
    data_folder=data.EXPERIMENT_FOLDER,
    save_folder=data.TEMP_FOLDER
):
    graph_padding = 0.1

    power_correlation_data = get.get_reward_correlations_and_powers_from_sweep(
        sweep_id, include_baseline_power=True, folder=data_folder
    )

    reward_correlations, all_powers_A, all_powers_B, baseline_powers_A = (
        power_correlation_data['reward_correlations'],
        power_correlation_data['all_powers_A'],
        power_correlation_data['all_powers_B'],
        power_correlation_data['baseline_powers_A']
    )

    min_A_power = min([powers_A.min() for powers_A in all_powers_A]) * (1 - graph_padding)
    max_A_power = max([powers_A.max() for powers_A in all_powers_A]) * (1 + graph_padding)
    min_B_power = min([powers_B.min() for powers_B in all_powers_B]) * (1 - graph_padding)
    max_B_power = max([powers_B.max() for powers_B in all_powers_B]) * (1 + graph_padding)

    power_correlations_ = []
    all_fig_names_ = []

    for correlation, powers_A, powers_B in zip(reward_correlations, all_powers_A, all_powers_B):
        fig, ax = plt.subplots()
        current_fig_name = '{0}-sweep_id_{1}-specific_alignment_curve-correlation_{2}'.format(
            fig_name, sweep_id, str(correlation)
        )

        if include_baseline_powers:
            for baseline_power in baseline_powers_A:
                ax.plot([baseline_power.item()] * 2, [min_B_power, max_B_power], 'b-', alpha=0.25)

        ax.plot(powers_A, powers_B, 'mh')

        ax.set_xlabel('State POWER values, Agent A')
        ax.set_ylabel('State POWER values, Agent B')
        ax.set_title('Reward correlation value {}'.format(correlation))
        ax.set_xlim([min_A_power, max_A_power])
        ax.set_ylim([min_B_power, max_B_power])

        data.save_figure(fig, current_fig_name, folder=save_folder)

        power_correlations_ += [torch.corrcoef(torch.stack([powers_A, powers_B]))[0][1]]
        all_fig_names_ += [current_fig_name]

    anim.animate_from_filenames(
        all_fig_names_,
        'specific_power_alignment_animation',
        ms_per_frame=ms_per_frame,
        input_folder_or_list=save_folder,
        output_folder=save_folder
    )

    plot_power_correlation_relationship(
        reward_correlations,
        all_powers_A,
        all_powers_B,
        x_axis_label='Reward correlation value',
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
    show=True,
    fig_name=None,
    distribution_dict=dist.DISTRIBUTION_DICT,
    save_folder=data.TEMP_FOLDER
):
    if random_seed:
        misc.set_global_random_seed(random_seed)

    agent_A_dist = dist.config_to_reward_distribution(
        ['TEMP'], { 'default_dist': distribution_config }, distribution_dict=distribution_dict
    )
    agent_A_samples = agent_A_dist(num_samples)
    agent_B_samples = dist.generate_correlated_reward_samples(
        agent_A_dist, agent_A_samples, correlation=correlation, symmetric_interval=symmetric_interval
    )

    fig, ax = plt.subplots()

    ax.plot(np.array(agent_A_samples.T[0]), np.array(agent_B_samples.T[0]), 'mh')
    ax.set_xlabel('Agent A reward samples')
    ax.set_ylabel('Agent B reward samples')
    ax.set_title('Correlation = {:.2f}'.format(correlation))

    if fig_name is not None:
        data.save_figure(fig, fig_name, folder=save_folder)

    if show:
        plt.show()