import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import torch
import math

from . import graph

def calculate_sample_aggregations(all_samples, aggregation='mean'):
    if aggregation == 'mean':
        return all_samples.mean(axis=0)
    elif aggregation == 'var':
        return all_samples.var(axis=0)

def generate_fig_layout(subplots, sharey=True):
    # If layout is for a gridworld, subplots will be a 2-tuple
    if isinstance(subplots, tuple) and len(subplots) == 2:
        fig_rows, fig_cols = subplots
    else:
        fig_cols = min(subplots, 4)
        fig_rows = math.ceil(subplots / fig_cols)

    fig, axs = plt.subplots(
        fig_rows,
        fig_cols,
        sharex=True,
        sharey=sharey,
        tight_layout=True,
        figsize=(4 * fig_cols, 4 * fig_rows)
    )

    axs_rows = axs if fig_cols > 1 else [axs]

    return (
        fig_cols,
        fig_rows,
        fig,
        axs_rows if fig_rows > 1 else [axs_rows]
    )

def generate_gridworld_fig_data(plotted_states, sharey=False):
    row_coords, col_coords = np.array(graph.gridworld_states_to_coords(plotted_states)).T

    _, fig_rows, fig_, axs_plot_ = generate_fig_layout(
        (max(row_coords) + 1, max(col_coords) + 1), sharey=sharey
    )
    return [
        fig_rows,
        fig_,
        axs_plot_,
        list(zip(row_coords, col_coords))
    ]

def generate_standard_fig_data(state_indices, sharey=False):
    fig_cols, fig_rows, fig_, axs_plot_ = generate_fig_layout(len(state_indices), sharey=sharey)

    return [
        fig_rows,
        fig_,
        axs_plot_,
        [(i // fig_cols, i % fig_cols) for i in range(len(state_indices))]
    ]

def render_gridworld_aggregations(
    all_samples,
    state_list,
    aggregation='mean',
    sample_quantity='POWER',
    sample_units='reward units'
):
    sample_aggregations = calculate_sample_aggregations(all_samples, aggregation=aggregation)

    row_coords, col_coords = np.array(graph.gridworld_states_to_coords(state_list)).T
    num_rows, num_cols = max(row_coords) + 1, max(col_coords) + 1
    excluded_coords = list(
        set(
            it.product(range(num_rows), range(num_cols))
        ) - set(
            [(row_coord, col_coord) for row_coord, col_coord in zip(row_coords, col_coords)]
        )
    )

    fig, ax_ = plt.subplots(figsize=(num_rows, num_cols), tight_layout=True)

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
    
    return fig

def render_standard_aggregations(
    all_samples,
    state_list,
    aggregation='mean',
    sample_quantity='POWER',
    sample_units='reward units'
):
    sample_aggregations = calculate_sample_aggregations(all_samples, aggregation=aggregation)

    fig, ax_ = plt.subplots(tight_layout=True)

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

    return fig

def render_distributions(
    all_samples,
    plotted_states,
    state_indices,
    plot_as_gridworld=False,
    number_of_bins=30,
    normalize_auc=True,
    sample_quantity='POWER',
    sample_units='reward units'
):
    transposed_samples = torch.transpose(all_samples, 0, 1)

    if plot_as_gridworld:
        fig_rows, fig_, axs_plot_, axis_coords_list = generate_gridworld_fig_data(plotted_states, sharey=True)
    else:
        fig_rows, fig_, axs_plot_, axis_coords_list = generate_standard_fig_data(state_indices, sharey=True)

    for axis_coords, state, state_index in zip(axis_coords_list, plotted_states, state_indices):
        # The hist function hangs unless we convert to numpy first
        axs_plot_[axis_coords[0]][axis_coords[1]].hist(
            np.array(transposed_samples[state_index]),
            bins=np.linspace(
                transposed_samples.min(), transposed_samples.max(), number_of_bins
            ) if normalize_auc else number_of_bins
        )
        axs_plot_[axis_coords[0]][axis_coords[1]].title.set_text(state)
    
    fig_.text(0.5, 0.01 / fig_rows, '{0} samples ({1})'.format(sample_quantity, sample_units))

    return fig_

def render_correlations(
    all_samples,
    state_x,
    state_x_index,
    state_y_list,
    state_y_indices,
    plot_as_gridworld=False,
    number_of_bins=30,
    sample_quantity='POWER',
    sample_units='reward_units'
):
    if plot_as_gridworld:
        fig_rows, fig_, axs_plot_, axis_coords_list = generate_gridworld_fig_data(state_y_list, sharey=False)
    else:
        fig_rows, fig_, axs_plot_, axis_coords_list = generate_standard_fig_data(state_y_indices, sharey=False)

    for axis_coords, state_y, state_y_index in zip(axis_coords_list, state_y_list, state_y_indices):
        axs_plot_[axis_coords[0]][axis_coords[1]].hist2d(
            np.array(torch.transpose(all_samples, 0, 1)[state_x_index]),
            np.array(torch.transpose(all_samples, 0, 1)[state_y_index]),
            bins=number_of_bins
        )
        axs_plot_[axis_coords[0]][axis_coords[1]].set_ylabel(
            '{0} sample of state {1} ({2})'.format(sample_quantity, state_y, sample_units)
        )

    fig_.text(0.5, 0.01 / fig_rows, '{0} sample of state {1} ({2})'.format(sample_quantity, state_x, sample_units))

    return fig_