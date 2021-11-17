import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb as wb

import data
import utils

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

def plot_sample_means(
    all_samples,
    state_list,
    show=True,
    save_handle=None,
    sample_quantity='POWER',
    sample_units='reward units',
    save_figure=data.save_figure,
    save_folder=data.EXPERIMENT_FOLDER
):
    fig, ax = plt.subplots()

    ax.bar(
        range(len(state_list)),
        torch.mean(all_samples, axis=0),
        yerr=torch.std(all_samples, axis=0) / np.sqrt(len(all_samples)),
        align='center',
        ecolor='black',
        capsize=10
    )
    ax.set_ylabel('{0} of state ({1})'.format(sample_quantity, sample_units))
    ax.set_xticks(range(len(state_list)))
    ax.set_xticklabels(state_list)
    ax.set_title('{} (mean $\pm$ standard error of the mean) for each state'.format(sample_quantity))

    plt.tight_layout()

    if callable(save_figure):
        save_figure(fig, '{0}_means-{1}'.format(sample_quantity, save_handle), folder=save_folder)

    if show:
        plt.show()
    
    return fig

def plot_sample_distributions(
    all_samples,
    state_list,
    states_to_plot=None,
    number_of_bins=30,
    normalize_auc=True,
    show=True,
    sample_quantity='POWER',
    sample_units='reward units',
    save_handle=None,
    save_figure=data.save_figure,
    save_folder=data.EXPERIMENT_FOLDER
):
    # The terminal state (last in the list) has power 0 in all samples, so we don't plot it by default.
    plotted_states = state_list[:-1] if (states_to_plot is None) else states_to_plot
    state_indices = [state_list.index(state_label) for state_label in plotted_states]

    fig_cols, fig_rows, fig, axs_plot = utils.generate_fig_layout(len(state_indices))

    transposed_samples = torch.transpose(all_samples, 0, 1)

    for i in range(len(state_indices)):
        # The hist function hangs unless we convert to numpy first
        axs_plot[i // fig_cols][i % fig_cols].hist(
            np.array(transposed_samples[state_indices[i]]),
            bins=np.linspace(
                (transposed_samples[:-1]).min(), (transposed_samples[:-1]).max(), number_of_bins
            ) if normalize_auc else number_of_bins
        )
        axs_plot[i // fig_cols][i % fig_cols].title.set_text(plotted_states[i])
    
    fig.text(0.5, 0.01 / fig_rows, '{0} samples ({1})'.format(sample_quantity, sample_units))

    if callable(save_figure):
        save_figure(fig, '{0}_samples-{1}'.format(sample_quantity, save_handle), folder=save_folder)
    
    if show:
        plt.show()
    
    return fig

def plot_sample_correlations(
    all_samples,
    state_list,
    state_x,
    state_list_y=None,
    number_of_bins=30,
    show=True,
    sample_quantity='POWER',
    sample_units='reward units',
    save_handle=None,
    save_figure=data.save_figure,
    save_folder=data.EXPERIMENT_FOLDER
):
    # The terminal state (last in the list) has power 0 in all samples, so we don't plot it by default.
    state_y_list = state_list[:-1] if (state_list_y is None) else state_list_y
    state_x_index = state_list.index(state_x)
    state_y_indices = [state_list.index(state_label) for state_label in state_y_list]

    fig_cols, fig_rows, fig, axs_plot = utils.generate_fig_layout(len(state_y_indices))

    for i in range(len(state_y_indices)):
        axs_plot[i // fig_cols][i % fig_cols].hist2d(
            np.array(torch.transpose(all_samples, 0, 1)[state_x_index]),
            np.array(torch.transpose(all_samples, 0, 1)[state_y_indices[i]]),
            bins=number_of_bins
        )
        axs_plot[i // fig_cols][i % fig_cols].set_ylabel(
            '{0} sample of state {1} ({2})'.format(sample_quantity, state_y_list[i], sample_units)
        )

    fig.text(0.5, 0.01 / fig_rows, '{0} sample of state {1} ({2})'.format(sample_quantity, state_x, sample_units))

    if callable(save_figure):
        save_figure(
            fig, '{0}_correlations_{1}-{2}'.format(sample_quantity, state_x, save_handle), folder=save_folder
        )

    if show:
        plt.show()
    
    return fig

# Here sample_filter is, e.g., reward_samples[:, 3] < reward_samples[:, 4]
def render_all_outputs(reward_samples, power_samples, state_list, sample_filter=None, **kwargs):
    rs_inputs = reward_samples if sample_filter is None else reward_samples[sample_filter]
    ps_inputs = power_samples if sample_filter is None else power_samples[sample_filter]

    print()
    print('Rendering plots...')

    return {
        'POWER means': plot_sample_means(
            ps_inputs, state_list, sample_quantity='POWER', sample_units='reward units', **kwargs
        ),
        'Reward samples over states': plot_sample_distributions(
            rs_inputs, state_list, sample_quantity='Reward', sample_units='reward units', **kwargs
        ),
        'POWER samples over states': plot_sample_distributions(
            ps_inputs, state_list, sample_quantity='POWER', sample_units='reward units', **kwargs
        ),
        **{
            'POWER correlations, state {}'.format(state): plot_sample_correlations(
                ps_inputs, state_list, state, sample_quantity='POWER', sample_units='reward units', **kwargs
            ) for state in state_list[:-1] # Don't plot or save terminal state
        }
    }
