import matplotlib.pyplot as plt
import numpy as np
import torch
import itertools as it

import data
import utils

def get_mean_values(state_samples, state_list):
    return { state_list[i]: torch.mean(state_samples, axis=0)[i] for i in range(len(state_list)) }

def plot_sample_means(
    all_samples,
    state_list,
    show=True,
    plot_as_gridworld=False,
    save_handle=None,
    sample_quantity='POWER',
    sample_units='reward units',
    save_figure=data.save_figure,
    save_folder=data.EXPERIMENT_FOLDER
):
    fig, ax_ = plt.subplots()

    if plot_as_gridworld:
        row_coords, col_coords = utils.gridworld_coords_from_states(state_list)
        excluded_coords = list(
            set(
                it.product(range(max(row_coords) + 1), range(max(col_coords) + 1))
            ) - set(
                [(row_coord, col_coord) for row_coord, col_coord in zip(row_coords, col_coords)]
            )
        )
        # Fill excluded coords with nan values to maximize contrast for non-nan entries
        heat_map, _, _ = np.histogram2d(
            np.append(row_coords, [coords[0] for coords in excluded_coords]),
            np.append(col_coords, [coords[1] for coords in excluded_coords]),
            bins=[max(row_coords) + 1, max(col_coords) + 1],
        # Don't plot TERMINAL
            weights=np.append(all_samples.mean(axis=0)[:-1], np.full(len(excluded_coords), np.nan))
        )

        ax_.imshow(heat_map)
        ax_.set_xticks(range(max(col_coords) + 1))
        ax_.set_yticks(range(max(row_coords) + 1))
        ax_.set_title('{0} means for each gridworld state ({1})'.format(sample_quantity, sample_units))

    else:
        ax_.bar(
            range(len(state_list)),
            torch.mean(all_samples, axis=0),
            yerr=torch.std(all_samples, axis=0) / np.sqrt(len(all_samples)),
            align='center',
            ecolor='black',
            capsize=10
        )
        ax_.set_ylabel('{0} of state ({1})'.format(sample_quantity, sample_units))
        ax_.set_xticks(range(len(state_list)))
        ax_.set_xticklabels(state_list)
        ax_.set_title('{} (mean $\pm$ standard error of the mean) for each state'.format(sample_quantity))

    plt.tight_layout()

    if callable(save_figure) and save_handle is not None:
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
    plot_as_gridworld=False,
    sample_quantity='POWER',
    sample_units='reward units',
    save_handle=None,
    save_figure=data.save_figure,
    save_folder=data.EXPERIMENT_FOLDER
):
    # The terminal state (last in the list) has power 0 in all samples, so we don't plot it by default.
    plotted_states = state_list[:-1] if (states_to_plot is None) else states_to_plot
    state_indices = [state_list.index(state_label) for state_label in plotted_states]
    transposed_samples = torch.transpose(all_samples, 0, 1)

    if plot_as_gridworld:
        row_coords, col_coords = utils.gridworld_coords_from_states(plotted_states)

        fig_cols, fig_rows, fig, axs_plot_ = utils.generate_fig_layout(
            (max(row_coords) + 1, max(col_coords) + 1), sharey=True
        )
        axis_coords_list = list(zip(row_coords, col_coords))
    
    else:
        fig_cols, fig_rows, fig, axs_plot_ = utils.generate_fig_layout(len(state_indices), sharey=True)
        axis_coords_list = [(i // fig_cols, i % fig_cols) for i in range(len(state_indices))]

    for axis_coords, state, state_index in zip(axis_coords_list, plotted_states, state_indices):
        # The hist function hangs unless we convert to numpy first
        axs_plot_[axis_coords[0]][axis_coords[1]].hist(
            np.array(transposed_samples[state_index]),
            bins=np.linspace(
                (transposed_samples[:-1]).min(), (transposed_samples[:-1]).max(), number_of_bins
            ) if normalize_auc else number_of_bins
        )
        axs_plot_[axis_coords[0]][axis_coords[1]].title.set_text(state)
    
    fig.text(0.5, 0.01 / fig_rows, '{0} samples ({1})'.format(sample_quantity, sample_units))

    if callable(save_figure) and save_handle is not None:
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
    plot_as_gridworld=False,
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

    if plot_as_gridworld:
        row_coords, col_coords = utils.gridworld_coords_from_states(state_y_list)

        fig_cols, fig_rows, fig, axs_plot_ = utils.generate_fig_layout(
            (max(row_coords) + 1, max(col_coords) + 1), sharey=False
        )
        axis_coords_list = list(zip(row_coords, col_coords))

    else:
        fig_cols, fig_rows, fig, axs_plot_ = utils.generate_fig_layout(len(state_y_indices), sharey=False)
        axis_coords_list = [(i // fig_cols, i % fig_cols) for i in range(len(state_y_indices))]

    for axis_coords, state_y, state_y_index in zip(axis_coords_list, state_y_list, state_y_indices):
        axs_plot_[axis_coords[0]][axis_coords[1]].hist2d(
            np.array(torch.transpose(all_samples, 0, 1)[state_x_index]),
            np.array(torch.transpose(all_samples, 0, 1)[state_y_index]),
            bins=number_of_bins
        )
        axs_plot_[axis_coords[0]][axis_coords[1]].set_ylabel(
            '{0} sample of state {1} ({2})'.format(sample_quantity, state_y, sample_units)
        )

    fig.text(0.5, 0.01 / fig_rows, '{0} sample of state {1} ({2})'.format(sample_quantity, state_x, sample_units))

    if callable(save_figure) and save_handle is not None:
        save_figure(
            fig, '{0}_correlations_{1}-{2}'.format(sample_quantity, state_x, save_handle), folder=save_folder
        )

    if show:
        plt.show()
    
    return fig

def plot_mdp_graph(
    mdp_graph,
    show=True,
    save_handle='temp',
    save_folder=data.TEMP_FOLDER,
    temp_folder=data.TEMP_FOLDER
):
# We save to temp solely for the purpose of plotting, since Graphviz prefers to consume files
# rather than literal dot strings. We save it in temp so as not to overwrite "permanent" MDP graphs
# and so git doesn't track these.
    data.save_graph_to_dot_file(mdp_graph, save_handle, folder=temp_folder)
    fig = data.create_and_save_mdp_figure(
        save_handle,
        fig_name='mdp_graph-{}'.format(save_handle),
        mdps_folder=temp_folder,
        fig_folder=save_folder,
        show=show
    )

    return fig

# Here sample_filter is, e.g., reward_samples[:, 3] < reward_samples[:, 4]
def render_all_outputs(reward_samples, power_samples, mdp_graph, sample_filter=None, **kwargs):
    state_list = list(mdp_graph)
    rs_inputs = reward_samples if sample_filter is None else reward_samples[sample_filter]
    ps_inputs = power_samples if sample_filter is None else power_samples[sample_filter]
    mdp_kwargs = {
        key: kwargs[key] for key in list(
            set(kwargs.keys()) & set(['show', 'save_handle', 'save_folder', 'temp_folder'])
        )
    }

    print()
    print('Rendering plots...')

    return {
        'MDP graph': plot_mdp_graph(mdp_graph, **mdp_kwargs),
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
