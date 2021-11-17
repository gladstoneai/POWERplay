import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb as wb
import math

import data

def policy_rollout(policy, reward_function, state_list, starting_state, number_of_steps):
    if len(policy) != len(reward_function):
        raise Exception('The policy must have the same number of rows as the reward_function has entries.')
    
    if len(reward_function) != len(state_list):
        raise Exception('The reward_function and state_list must have the same length.')   

    state_index = state_list.index(starting_state)

    state_history = [starting_state]
    reward_history = [reward_function[state_index].item()]

    for _ in range(number_of_steps):
        state_index = (policy[state_index] == 1).nonzero()[0].item()

        state_history += [state_list[state_index]]
        reward_history += [reward_function[state_index].item()]
    
    return (
        state_history, reward_history
    )

def get_mean_values(state_samples, state_list):
    return { state_list[i]: torch.mean(state_samples, axis=0)[i] for i in range(len(state_list)) }

def plot_power_means(
    power_samples,
    state_list,
    show=True,
    save_fig=False,
    save_handle=None,
    save_folder=data.EXPERIMENT_FOLDER,
    wb_tracker=None
):
    fig, ax = plt.subplots()

    ax.bar(
        range(len(state_list)),
        torch.mean(power_samples, axis=0),
        yerr=torch.std(power_samples, axis=0) / np.sqrt(len(power_samples)),
        align='center',
        ecolor='black',
        capsize=10
    )
    ax.set_ylabel('POWER of state (reward units)')
    ax.set_xticks(range(len(state_list)))
    ax.set_xticklabels(state_list)
    ax.set_title('POWER (mean $\pm$ standard error of the mean) for each state')

    plt.tight_layout()

    if save_fig:
        data.save_figure(fig, 'power_means-{}'.format(save_handle), folder=save_folder)

    if wb_tracker is not None:
        wb_tracker.log({ 'POWER means': wb.Image(fig) })
    
    if show:
        plt.show()

def plot_power_samples(
    power_samples,
    state_list,
    states_to_plot=None,
    number_of_bins=30,
    normalize_auc=True,
    show=True,
    save_fig=False,
    save_handle=None,
    common_x_axis='POWER sample (reward units)',
    save_fig_prefix='power_samples',
    save_folder=data.EXPERIMENT_FOLDER,
    wb_tracker=None
):
    # The terminal state (last in the list) has power 0 in all samples, so we don't plot it by default.
    plotted_states = state_list[:-1] if (states_to_plot is None) else states_to_plot
    state_indices = [state_list.index(state_label) for state_label in plotted_states]

    fig_cols = min(len(state_indices), 4)
    fig_rows = math.ceil(len(state_indices) / fig_cols)

    fig, axs = plt.subplots(
        fig_rows,
        fig_cols,
        sharex=True,
        sharey=True,
        tight_layout=True,
        figsize=(4 * fig_cols, 4 * fig_rows)
    )

    axs_rows = axs if fig_cols > 1 else [axs]
    axs_plot = axs_rows if fig_rows > 1 else [axs_rows]
    transposed_samples = torch.transpose(power_samples, 0, 1)

    for i in range(len(state_indices)):
        # The hist function hangs unless we convert to numpy first
        axs_plot[i // fig_cols][i % fig_cols].hist(
            np.array(transposed_samples[state_indices[i]]),
            bins=np.linspace(
                (transposed_samples[:-1]).min(), (transposed_samples[:-1]).max(), number_of_bins
            ) if normalize_auc else number_of_bins
        )
        axs_plot[i // fig_cols][i % fig_cols].title.set_text(plotted_states[i])
    
    fig.text(0.5, 0.01 / fig_rows, common_x_axis)

    if save_fig:
        data.save_figure(fig, '{0}-{1}'.format(save_fig_prefix, save_handle), folder=save_folder)
    
    if wb_tracker is not None:
        wb_tracker.log({ 'Distributions over states': wb.Image(fig) })
    
    if show:
        plt.show()

def plot_power_correlations(
    power_samples,
    state_list,
    state_x,
    state_list_y=None,
    number_of_bins=30,
    show=True,
    save_fig=False,
    save_handle=None,
    save_folder=data.EXPERIMENT_FOLDER,
    wb_tracker=None
):
    # The terminal state (last in the list) has power 0 in all samples, so we don't plot it by default.
    state_y_list = state_list[:-1] if (state_list_y is None) else state_list_y
    state_x_index = state_list.index(state_x)
    state_y_indices = [state_list.index(state_label) for state_label in state_y_list]

    fig_cols = min(len(state_y_indices), 4)
    fig_rows = math.ceil(len(state_y_indices) / fig_cols)

    fig, axs = plt.subplots(
        fig_rows,
        fig_cols,
        sharex=True,
        sharey=False,
        tight_layout=True,
        figsize=(4 * fig_cols, 4 * fig_rows)
    )

    axs_rows = axs if fig_cols > 1 else [axs]
    axs_plot = axs_rows if fig_rows > 1 else [axs_rows]

    for i in range(len(state_y_indices)):
        axs_plot[i // fig_cols][i % fig_cols].hist2d(
            np.array(torch.transpose(power_samples, 0, 1)[state_x_index]),
            np.array(torch.transpose(power_samples, 0, 1)[state_y_indices[i]]),
            bins=number_of_bins
        )
        axs_plot[i // fig_cols][i % fig_cols].set_ylabel(
            'POWER sample of state {} (reward units)'.format(state_y_list[i])
        )

    fig.text(0.5, 0.01 / fig_rows, 'POWER sample of state {} (reward units)'.format(state_x))

    if save_fig:
        data.save_figure(
            fig, 'power_correlations_{0}-{1}'.format(state_x, save_handle), folder=save_folder
        )
    
    if wb_tracker is not None:
        wb_tracker.log({ 'POWER correlations, state {}'.format(state_x): wb.Image(fig) })
    
    if show:
        plt.show()

def plot_reward_samples(reward_samples, state_list, **kwargs):
    plot_power_samples(
        reward_samples,
        state_list,
        common_x_axis='Reward sample (reward units)',
        save_fig_prefix='reward_samples',
        **kwargs
    )

# Here sample_filter is, e.g., reward_samples[:, 3] < reward_samples[:, 4]
def render_all_outputs(reward_samples, power_samples, state_list, sample_filter=None, **kwargs):
    rs_inputs = reward_samples if sample_filter is None else reward_samples[sample_filter]
    ps_inputs = power_samples if sample_filter is None else power_samples[sample_filter]

    print()
    print('Rendering plots...')

    plot_power_means(ps_inputs, state_list, **kwargs)
    plot_reward_samples(rs_inputs, state_list, **kwargs)
    plot_power_samples(ps_inputs, state_list, **kwargs)
    
    for state in state_list[:-1]: # Don't plot or save terminal state
        plot_power_correlations(ps_inputs, state_list, state, **kwargs)
