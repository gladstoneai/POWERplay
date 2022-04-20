import matplotlib.pyplot as plt
import numpy as np
import torch
import itertools as it

from .utils import graph
from .utils import misc
from . import data

def plot_sample_aggregations(
    all_samples,
    state_list,
    show=True,
    plot_as_gridworld=False,
    save_handle=None,
    aggregation='mean',
    sample_quantity='POWER',
    sample_units='reward units',
    save_figure=data.save_figure,
    save_folder=data.EXPERIMENT_FOLDER
):
    if aggregation == 'mean':
        sample_aggregations = all_samples.mean(axis=0)
    elif aggregation == 'var':
        sample_aggregations = all_samples.var(axis=0)
    
    if plot_as_gridworld:
        row_coords, col_coords = np.array(graph.gridworld_states_to_coords(state_list)).T
        num_rows, num_cols = max(row_coords) + 1, max(col_coords) + 1
        excluded_coords = list(
            set(
                it.product(range(num_rows), range(num_cols))
            ) - set(
                [(row_coord, col_coord) for row_coord, col_coord in zip(row_coords, col_coords)]
            )
        )

        fig, ax_ = plt.subplots(figsize=(num_rows, num_cols))

        # Fill excluded coords with nan values to maximize contrast for non-nan entries
        heat_map, _, _ = np.histogram2d(
            np.append(row_coords, [coords[0] for coords in excluded_coords]),
            np.append(col_coords, [coords[1] for coords in excluded_coords]),
            bins=[num_rows, num_cols],
            weights=np.append(sample_aggregations, np.full(len(excluded_coords), np.nan))
        )

        ax_.imshow(heat_map)
        ax_.set_xticks(range(num_cols))
        ax_.set_yticks(range(num_rows))
        ax_.set_title('{0} {1}s for each gridworld state ({2})'.format(sample_quantity, aggregation, sample_units))

        for sample_index in range(len(row_coords)):
            ax_.text(
                col_coords[sample_index],
                row_coords[sample_index],
                round(float(sample_aggregations[sample_index]), 4),
                ha='center',
                va='center',
                color='w'
            )

    else:
        fig, ax_ = plt.subplots()

        bars = ax_.bar(
            range(len(state_list)),
            sample_aggregations,
            yerr=torch.std(all_samples, axis=0) / np.sqrt(len(all_samples)) if aggregation == 'mean' else None,
            align='center',
            ecolor='black',
            capsize=10
        )
        ax_.set_ylabel('{0} of state ({1})'.format(sample_quantity, sample_units))
        ax_.set_xticks(range(len(state_list)))
        ax_.set_xticklabels(state_list, rotation='vertical')
        ax_.set_title('{0} ({1}{2}) for each state'.format(
            sample_quantity, aggregation, ' $\pm$ standard error of the mean' if aggregation == 'mean' else ''
        ))
        ax_.bar_label(bars, rotation='vertical', label_type='center')

    plt.tight_layout()

    if callable(save_figure) and save_handle is not None:
        save_figure(fig, '{0}_{1}s-{2}'.format(sample_quantity, aggregation, save_handle), folder=save_folder)

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
    plotted_states = state_list if (states_to_plot is None) else states_to_plot
    state_indices = [state_list.index(state_label) for state_label in plotted_states]
    transposed_samples = torch.transpose(all_samples, 0, 1)

    if plot_as_gridworld:
        row_coords, col_coords = np.array(graph.gridworld_states_to_coords(plotted_states)).T

        fig_cols, fig_rows, fig, axs_plot_ = misc.generate_fig_layout(
            (max(row_coords) + 1, max(col_coords) + 1), sharey=True
        )
        axis_coords_list = list(zip(row_coords, col_coords))
    
    else:
        fig_cols, fig_rows, fig, axs_plot_ = misc.generate_fig_layout(len(state_indices), sharey=True)
        axis_coords_list = [(i // fig_cols, i % fig_cols) for i in range(len(state_indices))]

    for axis_coords, state, state_index in zip(axis_coords_list, plotted_states, state_indices):
        # The hist function hangs unless we convert to numpy first
        axs_plot_[axis_coords[0]][axis_coords[1]].hist(
            np.array(transposed_samples[state_index]),
            bins=np.linspace(
                (transposed_samples[:-1]).min(), (transposed_samples[:-1]).max(), number_of_bins # HERE: Why do we not look at the last sample?
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
    state_y_list = state_list if (state_list_y is None) else state_list_y
    state_x_index = state_list.index(state_x)
    state_y_indices = [state_list.index(state_label) for state_label in state_y_list]

    if plot_as_gridworld:
        row_coords, col_coords = np.array(graph.gridworld_states_to_coords(state_y_list)).T

        fig_cols, fig_rows, fig, axs_plot_ = misc.generate_fig_layout(
            (max(row_coords) + 1, max(col_coords) + 1), sharey=False
        )
        axis_coords_list = list(zip(row_coords, col_coords))

    else:
        fig_cols, fig_rows, fig, axs_plot_ = misc.generate_fig_layout(len(state_y_indices), sharey=False)
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

def plot_mdp_or_policy(
    mdp_or_policy_graph,
    show=True,
    subgraphs_per_row=4,
    save_handle='temp',
    graph_type='mdp_graph',
    save_folder=data.TEMP_FOLDER,
    temp_folder=data.TEMP_FOLDER
):

    graph_to_plot = graph.transform_graph_for_plots(mdp_or_policy_graph)

# We save to temp solely for the purpose of plotting, since Graphviz prefers to consume files
# rather than literal dot strings. We save it in temp so as not to overwrite "permanent" MDP graphs
# and so git doesn't track these.
    data.save_graph_to_dot_file(graph_to_plot, save_handle, folder=temp_folder)
    fig = data.create_and_save_mdp_figure(
        save_handle,
        subgraphs_per_row=subgraphs_per_row,
        fig_name='{0}-{1}'.format(graph_type, save_handle),
        mdps_folder=temp_folder,
        fig_folder=save_folder,
        show=show
    )

    return fig

# Here sample_filter is, e.g., reward_samples[:, 3] < reward_samples[:, 4]
def render_all_outputs(
    reward_samples,
    power_samples,
    graphs_to_plot,
    sample_filter=None,
    plot_correlations=True,
    **kwargs
):
    state_list = graph.get_states_from_graph(graphs_to_plot[0]['graph_data'])
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
        'POWER means': plot_sample_aggregations(
            ps_inputs, state_list, aggregation='mean', sample_quantity='POWER', sample_units='reward units', **kwargs
        ),
        'POWER variances': plot_sample_aggregations(
            ps_inputs, state_list, aggregation='var', sample_quantity='POWER', sample_units='reward units', **kwargs
        ),
        'Reward samples over states': plot_sample_distributions(
            rs_inputs, state_list, sample_quantity='Reward', sample_units='reward units', **kwargs
        ),
        'POWER samples over states': plot_sample_distributions(
            ps_inputs, state_list, sample_quantity='POWER', sample_units='reward units', **kwargs
        ),
        **({
            'POWER correlations, state {}'.format(state): plot_sample_correlations(
                ps_inputs, state_list, state, sample_quantity='POWER', sample_units='reward units', **kwargs
            ) for state in state_list
        } if plot_correlations else {}),
        **{
            graph_to_plot['graph_name']: plot_mdp_or_policy(
                graph_to_plot['graph_data'], **{ **mdp_kwargs, 'graph_type': graph_to_plot['graph_type'] }
            ) for graph_to_plot in graphs_to_plot
        }
    }
