from matplotlib import rcParams

import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import torch
import math

from . import graph

################################################################################

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Avenir'] + rcParams['font.sans-serif']

FIGURE_SIZE_SCALING_FACTOR = 4

################################################################################

def calculate_sample_aggregations(all_samples, aggregation='mean'):
    if aggregation == 'mean':
        return all_samples.mean(axis=0)
    elif aggregation == 'var':
        return all_samples.var(axis=0)

def generate_fig_layout(
    subplots,
    sharey=True,
    figsize=None,
    scaling_factor=FIGURE_SIZE_SCALING_FACTOR
):
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
        figsize=(scaling_factor * fig_cols, scaling_factor * fig_rows) if (figsize is None) else figsize
    )

    axs_rows = axs if fig_cols > 1 else [axs]

    return (
        fig_cols,
        fig_rows,
        fig,
        axs_rows if fig_rows > 1 else [axs_rows]
    )

def generate_gridworld_fig_data(
    states_to_subplot=[],
    states_per_subplot=[],
    sharey=False,
    scaling_factor=FIGURE_SIZE_SCALING_FACTOR
):
    axis_row_coords, axis_col_coords = (
        np.array(graph.gridworld_states_to_coords(states_to_subplot)).T
    ) if len(states_to_subplot) > 0 else np.array([[0, 0], [0, 0]])
    subplot_num_rows, subplot_num_cols = (
        np.max(np.array(graph.gridworld_states_to_coords(states_per_subplot)), axis=0) + 1
    ) if len(states_per_subplot) > 0 else [scaling_factor] * 2

    axis_num_rows, axis_num_cols = max(axis_row_coords) + 1, max(axis_col_coords) + 1

    _, fig_rows, fig_, axs_plot_ = generate_fig_layout(
        (axis_num_rows, axis_num_cols),
        sharey=sharey,
        figsize=(axis_num_cols * subplot_num_cols, axis_num_rows * subplot_num_rows)
    )

    return [
        fig_rows,
        fig_,
        axs_plot_,
        list(zip(axis_row_coords, axis_col_coords))
    ]

def generate_standard_fig_data(state_indices, sharey=False):
    fig_cols, fig_rows, fig_, axs_plot_ = generate_fig_layout(len(state_indices), sharey=sharey)

    return [
        fig_rows,
        fig_,
        axs_plot_,
        [(i // fig_cols, i % fig_cols) for i in range(len(state_indices))]
    ]

def render_gridworld_rollout_snapshot(
    state_list,
    current_state,
    reward_function=None,
    agent_whose_rewards_are_displayed='H'
):
    agent_A_default_state = '(0, 0)'
    agent_H_color = 'blue'
    agent_A_color = 'red'

    if graph.are_gridworld_states_multiagent(state_list):
        both_agent_states = np.array(graph.multiagent_states_to_single_agent_states(state_list)).T
        # When we want to visualize the Agent A rewards, we swap the state labels of the 2 agents (but we DON'T swap the
        # COLORS of the agents as they move around on the gridworld)
        agent_H_states, agent_A_states = both_agent_states if (
            agent_whose_rewards_are_displayed == 'H'
        ) else both_agent_states[::-1]

        agent_H_display_state, agent_A_display_state = graph.multiagent_state_to_single_agent_states(current_state)
        agent_A_current_state = agent_A_display_state if agent_whose_rewards_are_displayed == 'H' else agent_H_display_state
        
    else:
        agent_H_states, agent_A_states = np.array(state_list), np.full(len(state_list), agent_A_default_state)
        agent_H_display_state, agent_A_display_state = current_state, np.array([agent_A_default_state])
        agent_A_current_state = agent_A_display_state

    agent_H_unique_states = sorted(list(set(agent_H_states)))
    agent_H_rows, agent_H_cols = np.array(graph.gridworld_states_to_coords(agent_H_states)).T
    rewards_to_display = np.zeros(len(agent_H_states)) if reward_function is None else reward_function

    _, fig_, axs_plot_, _ = generate_gridworld_fig_data(
        states_per_subplot=agent_H_unique_states, states_to_subplot=np.array(['(0, 0)'])
    )

    num_rows, num_cols = max(agent_H_rows) + 1, max(agent_H_cols) + 1
    excluded_coords = list(
        set(
            it.product(range(num_rows), range(num_cols))
        ) - set(
            [(row_coord, col_coord) for row_coord, col_coord in zip(agent_H_rows, agent_H_cols)]
        )
    )

    vmin, vmax = rewards_to_display.min(), rewards_to_display.max()
    state_indices = np.where(agent_A_current_state == agent_A_states)[0]

    # Fill excluded coords with nan values to maximize contrast for non-nan entries
    heat_map, _, _ = np.histogram2d(
        np.append(agent_H_rows[state_indices], [coords[0] for coords in excluded_coords]),
        np.append(agent_H_cols[state_indices], [coords[1] for coords in excluded_coords]),
        bins=[num_rows, num_cols],
        weights=np.append(rewards_to_display[state_indices], np.full(len(excluded_coords), np.nan))
    )

    axs_plot_[0][0].imshow(heat_map, vmin=vmin, vmax=vmax)
    axs_plot_[0][0].set_xticks(range(num_cols))
    axs_plot_[0][0].set_yticks(range(num_rows))
    axs_plot_[0][0].set_title('Rewards for agent {0} ({1})'.format(
        agent_whose_rewards_are_displayed, agent_H_color if agent_whose_rewards_are_displayed == 'H' else agent_A_color)
    )

    if graph.are_gridworld_states_multiagent([current_state]):
        agent_H_coords, agent_A_coords = graph.gridworld_states_to_coords([agent_H_display_state, agent_A_display_state])
        axs_plot_[0][0].add_patch(
            pat.Rectangle((agent_H_coords[1] - 0.5, agent_H_coords[0] - 0.5), 1, 1, fill=False, edgecolor=agent_H_color, lw=12)
        )
        axs_plot_[0][0].add_patch(
            pat.Rectangle((agent_A_coords[1] - 0.5, agent_A_coords[0] - 0.5), 1, 1, fill=False, edgecolor=agent_A_color, lw=6)
        )
    
    for sample_index in state_indices:
        axs_plot_[0][0].text(
            agent_H_cols[sample_index],
            agent_H_rows[sample_index],
            round(float(rewards_to_display[sample_index]), 4),
            ha='center',
            va='center',
            color='w'
        )
    
    return fig_

def render_gridworld_aggregations(
    all_samples,
    state_list,
    aggregation='mean',
    sample_quantity='POWER',
    sample_units='reward units',
    manual_title=None
):
    agent_H_states, agent_A_states = np.array(
        graph.multiagent_states_to_single_agent_states(state_list)
    ).T if graph.are_gridworld_states_multiagent(state_list) else (
        np.array(state_list), np.array(['(0, 0)'] * len(state_list))
    )
    agent_H_unique_states, agent_A_unique_states = (
        sorted(list(set(agent_H_states))), sorted(list(set(agent_A_states)))
    )
    agent_H_rows, agent_H_cols = np.array(graph.gridworld_states_to_coords(agent_H_states)).T

    _, fig_, axs_plot_, axis_coords_list = generate_gridworld_fig_data(
        states_per_subplot=agent_H_unique_states, states_to_subplot=agent_A_unique_states, sharey=True
    )

    num_rows, num_cols = max(agent_H_rows) + 1, max(agent_H_cols) + 1
    excluded_coords = list(
        set(
            it.product(range(num_rows), range(num_cols))
        ) - set(
            [(row_coord, col_coord) for row_coord, col_coord in zip(agent_H_rows, agent_H_cols)]
        )
    )

    sample_aggregations = calculate_sample_aggregations(all_samples, aggregation=aggregation)
    vmin, vmax = sample_aggregations.min(), sample_aggregations.max()

    for axis_coords, agent_A_state, state_indices in zip(
        axis_coords_list, agent_A_unique_states, [
            np.where(agent_A_states == agent_A_unique_state)[0] for agent_A_unique_state in agent_A_unique_states
        ]
    ):
        # Fill excluded coords with nan values to maximize contrast for non-nan entries
        heat_map, _, _ = np.histogram2d(
            np.append(agent_H_rows[state_indices], [coords[0] for coords in excluded_coords]),
            np.append(agent_H_cols[state_indices], [coords[1] for coords in excluded_coords]),
            bins=[num_rows, num_cols],
            weights=np.append(sample_aggregations[state_indices], np.full(len(excluded_coords), np.nan))
        )

        axs_plot_[axis_coords[0]][axis_coords[1]].imshow(heat_map, vmin=vmin, vmax=vmax)
        axs_plot_[axis_coords[0]][axis_coords[1]].set_xticks(range(num_cols))
        axs_plot_[axis_coords[0]][axis_coords[1]].set_yticks(range(num_rows))

        if graph.are_gridworld_states_multiagent(state_list):
            axs_plot_[axis_coords[0]][axis_coords[1]].title.set_text('Agent A at {}'.format(agent_A_state))
            axs_plot_[axis_coords[0]][axis_coords[1]].add_patch(
                pat.Rectangle((axis_coords[1] - 0.5, axis_coords[0] - 0.5), 1, 1, fill=False, edgecolor='red', lw=3)
            )

        for sample_index in state_indices:
            axs_plot_[axis_coords[0]][axis_coords[1]].text(
                agent_H_cols[sample_index],
                agent_H_rows[sample_index],
                '{:.3f}'.format(float(sample_aggregations[sample_index])),
                ha='center',
                va='center',
                color='w',
                backgroundcolor='#00000025',
            )

    if manual_title is None:
        fig_.suptitle('{0} {1}s for each gridworld state ({2})'.format(sample_quantity, aggregation, sample_units))
    
    else:
        fig_.suptitle(manual_title)
        
    return fig_

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
    plot_in_log_scale=False,
    sample_quantity='POWER',
    sample_units='reward units'
):
    transposed_samples = torch.transpose(all_samples, 0, 1)

    agent_H_states, agent_A_states = np.array(
        graph.multiagent_states_to_single_agent_states(plotted_states)
    ).T if (plot_as_gridworld and graph.are_gridworld_states_multiagent(plotted_states)) else (
        np.array(plotted_states), np.array(['(0, 0)'] * len(plotted_states))
    )
    agent_A_unique_states = sorted(list(set(agent_A_states)))

    all_figs_ = []

    for agent_A_state in agent_A_unique_states:
        figure_indices = np.where(agent_A_states == agent_A_state)[0]

        if plot_as_gridworld:
            fig_rows, fig_, axs_plot_, axis_coords_list = generate_gridworld_fig_data(
                states_to_subplot=agent_H_states[figure_indices], sharey=True
            )

            if graph.are_gridworld_states_multiagent(plotted_states):
                agent_A_coords = graph.gridworld_state_to_coords(agent_A_state)

                fig_.suptitle('Agent A at {}'.format(agent_A_state))
                fig_.agent_A_state = agent_A_state

                axs_plot_[agent_A_coords[0]][agent_A_coords[1]].patch.set_edgecolor('red')
                axs_plot_[agent_A_coords[0]][agent_A_coords[1]].patch.set_linewidth(3)
        
        else:
            fig_rows, fig_, axs_plot_, axis_coords_list = generate_standard_fig_data(state_indices, sharey=True)

        for axis_coords, state, state_index in zip(
            axis_coords_list, agent_H_states[figure_indices], figure_indices
        ):
            bin_positions = np.logspace(np.log10(transposed_samples.min()), np.log10(transposed_samples.max()), number_of_bins) if (
                plot_in_log_scale
            ) else np.linspace(transposed_samples.min(), transposed_samples.max(), number_of_bins)

            # The hist function hangs unless we convert to numpy first
            axs_plot_[axis_coords[0]][axis_coords[1]].hist(
                np.array(transposed_samples[state_index]), bins=(bin_positions if normalize_auc else number_of_bins)
            )
            axs_plot_[axis_coords[0]][axis_coords[1]].title.set_text(state)

            if plot_in_log_scale:
                axs_plot_[axis_coords[0]][axis_coords[1]].set_xscale('log')

        fig_.text(0.5, 0.01 / fig_rows, '{0} samples ({1})'.format(sample_quantity, sample_units))

        all_figs_ += [fig_]

    return all_figs_

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
    agent_H_states, agent_A_states = np.array(
        graph.multiagent_states_to_single_agent_states(state_y_list)
    ).T if (plot_as_gridworld and graph.are_gridworld_states_multiagent(state_y_list)) else (
        np.array(state_y_list), np.array(['(0, 0)'] * len(state_y_list))
    )
    agent_A_unique_states = sorted(list(set(agent_A_states)))

    all_figs_ = []

    for agent_A_state in agent_A_unique_states:
        figure_indices = np.where(agent_A_states == agent_A_state)[0]

        if plot_as_gridworld:
            fig_rows, fig_, axs_plot_, axis_coords_list = generate_gridworld_fig_data(
                states_to_subplot=agent_H_states[figure_indices], sharey=False
            )

            if graph.are_gridworld_states_multiagent(state_y_list):
                agent_A_coords = graph.gridworld_state_to_coords(agent_A_state)

                fig_.suptitle('Agent A at {}'.format(agent_A_state))
                fig_.agent_A_state = agent_A_state

                axs_plot_[agent_A_coords[0]][agent_A_coords[1]].patch.set_edgecolor('red')
                axs_plot_[agent_A_coords[0]][agent_A_coords[1]].patch.set_linewidth(8)
        
        else:
            fig_rows, fig_, axs_plot_, axis_coords_list = generate_standard_fig_data(state_y_indices, sharey=False)

        for axis_coords, state_y, state_y_index in zip(
            axis_coords_list, np.array(state_y_list)[figure_indices], figure_indices
        ):
            # The hist function hangs unless we convert to numpy first
            axs_plot_[axis_coords[0]][axis_coords[1]].hist2d(
                np.array(torch.transpose(all_samples, 0, 1)[state_x_index]),
                np.array(torch.transpose(all_samples, 0, 1)[state_y_index]),
                bins=number_of_bins
            )
            axs_plot_[axis_coords[0]][axis_coords[1]].set_ylabel(
                '{0} sample of state {1} ({2})'.format(sample_quantity, state_y, sample_units)
            )

        fig_.text(0.5, 0.01 / fig_rows, '{0} sample of state {1} ({2})'.format(sample_quantity, state_x, sample_units))

        all_figs_ += [fig_]

    return all_figs_