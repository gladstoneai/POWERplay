import matplotlib.pyplot as plt
import collections as col

from .lib.utils import graph
from .lib.utils import render
from .lib import data
from . import anim
from . import policy

def plot_gridworld_rollout(
    state_list,
    state_rollout,
    reward_function=None,
    show=True,
    agent_whose_rewards_are_displayed='A',
    save_handle='gridworld_rollout',
    save_folder=data.TEMP_FOLDER
):
    all_figs_ = []
    for i in range(len(state_rollout)):
        all_figs_ += [render.render_gridworld_rollout_snapshot(
            state_list,
            state_rollout[i],
            reward_function=reward_function,
            agent_whose_rewards_are_displayed=agent_whose_rewards_are_displayed
        )]
        data.save_figure(all_figs_[-1], '{0}-{1}'.format(save_handle, i), folder=save_folder)

    anim.animate_rollout(
        save_handle,
        len(state_rollout),
        output_filename='{}-animation'.format(save_handle),
        input_folder=save_folder,
        output_folder=save_folder
    )

    if show:
        plt.show()

    return all_figs_

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
    if plot_as_gridworld:
        fig = render.render_gridworld_aggregations(
            all_samples,
            state_list,
            aggregation=aggregation,
            sample_quantity=sample_quantity,
            sample_units=sample_units
        )

    else:
        fig = render.render_standard_aggregations(
            all_samples,
            state_list,
            aggregation=aggregation,
            sample_quantity=sample_quantity,
            sample_units=sample_units
        )

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
    
    all_figs = render.render_distributions(
        all_samples,
        plotted_states,
        state_indices,
        plot_as_gridworld=plot_as_gridworld,
        number_of_bins=number_of_bins,
        normalize_auc=normalize_auc,
        sample_quantity=sample_quantity,
        sample_units=sample_units
    )

    if callable(save_figure) and save_handle is not None:
        for fig in all_figs:
            save_figure(fig, '{0}_samples{1}-{2}'.format(
                sample_quantity,
                '_agent_B_state_{}'.format(fig.agent_B_state) if (
                    hasattr(fig, 'agent_B_state')
                ) else '',
                save_handle
            ), folder=save_folder)
    
    if show:
        plt.show()
    
    return all_figs

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

    all_figs = render.render_correlations(
        all_samples,
        state_x,
        state_x_index,
        state_y_list,
        state_y_indices,
        plot_as_gridworld=plot_as_gridworld,
        number_of_bins=number_of_bins,
        sample_quantity=sample_quantity,
        sample_units=sample_units
    )

    if callable(save_figure) and save_handle is not None:
        for fig in all_figs:
            save_figure(
                fig, '{0}_correlations_{1}{2}-{3}'.format(
                    sample_quantity,
                    state_x,
                    '_agent_B_state{}'.format(fig.agent_B_state) if (
                        hasattr(fig, 'agent_B_state')
                    ) else '',
                    save_handle
                ), folder=save_folder
            )

    if show:
        plt.show()
    
    return all_figs

def plot_mdp_or_policy(
    mdp_or_policy_graph,
    show=True,
    subgraphs_per_row=4,
    reward_to_plot=None,
    discount_rate_to_plot=None,
    save_handle='temp',
    graph_type='mdp_graph_A',
    save_folder=data.TEMP_FOLDER,
    temp_folder=data.TEMP_FOLDER
):

    graph_to_plot = graph.transform_graph_for_plots(
        mdp_or_policy_graph, reward_to_plot=reward_to_plot, discount_rate_to_plot=discount_rate_to_plot
    )

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

# TODO: Document.
def plot_policy_sample(
    policy_graph,
    reward_function,
    discount_rate,
    show=True,
    subgraphs_per_row=4,
    save_handle='temp',
    save_folder=data.TEMP_FOLDER,
    temp_folder=data.TEMP_FOLDER,
):
    return plot_mdp_or_policy(
        policy_graph,
        show=show,
        subgraphs_per_row=subgraphs_per_row,
        reward_to_plot=reward_function,
        discount_rate_to_plot=discount_rate,
        save_handle=save_handle,
        graph_type='policy_reward_sample',
        save_folder=save_folder,
        temp_folder=temp_folder
    )

# NOTE: This function is not optimized for speed and runs very slowly, even for just 10 rollouts.
# It's also not capable of saving the plot and just shows it.
# run_properties: outupt of get.get_properties_from_run(sweep_id, run_suffix=run_suffix)
def plot_rollout_powers(run_properties, initial_state=None, number_of_rollouts=10, number_of_steps=20):
    state_list = graph.get_states_from_graph(run_properties['transition_graphs'][0])
    powers_A = run_properties['power_samples'].mean(dim=0)

    all_power_plot_rollouts_ = []

    for reward_sample_index in range(number_of_rollouts):

        print(reward_sample_index)
        policy_data = policy.sample_optimal_policy_from_run(run_properties, reward_sample_index=reward_sample_index)

        rollout = policy.simulate_policy_rollout(
            initial_state,
            policy_data['policy_graph_A'],
            policy_data['mdp_graph_A'],
            policy_graph_B=policy_data['policy_graph_B'],
            mdp_graph_B=policy_data['mdp_graph_B'],
            number_of_steps=number_of_steps
        )

        power_rollout = [powers_A[state_list.index(state)].item() for state in rollout]
        # In either multiagent case, we have to skip the rollout steps where Agent B moves
        power_plot_rollout = power_rollout if run_properties['sweep_type'] == 'single_agent' else power_rollout[::2]

        plt.plot(list(range(number_of_steps + 1)), power_plot_rollout, 'b-')
        all_power_plot_rollouts_ += [power_plot_rollout]

    plt.show()

    return all_power_plot_rollouts_

# Here sample_filter is, e.g., reward_samples[:, 3] < reward_samples[:, 4]
def plot_all_outputs(
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
        **{
            'Reward samples over states{}'.format(
                ', agent B at {}'.format(fig.agent_B_state) if hasattr(fig, 'agent_B_state') else ''
            ): fig for fig in plot_sample_distributions(
                rs_inputs, state_list, sample_quantity='Reward', sample_units='reward units', **kwargs
            )
        },
        **{
            'POWER samples over states{}'.format(
                ', agent B at {}'.format(fig.agent_B_state) if hasattr(fig, 'agent_B_state') else ''
            ): fig for fig in plot_sample_distributions(
                ps_inputs, state_list, sample_quantity='POWER', sample_units='reward units', **kwargs
            )
        },
        **(dict(col.ChainMap(*[{
                'POWER correlations, state {0}{1}'.format(
                    state, ', agent B at {}'.format(fig.agent_B_state) if hasattr(fig, 'agent_B_state') else ''
                ): fig for fig in plot_sample_correlations(
                    ps_inputs, state_list, state, sample_quantity='POWER', sample_units='reward units', **kwargs
                )
            } for state in state_list])) if plot_correlations else {}),
        **{
            graph_to_plot['graph_name']: plot_mdp_or_policy(
                graph_to_plot['graph_data'], **{ **mdp_kwargs, 'graph_type': graph_to_plot['graph_type'] }
            ) for graph_to_plot in graphs_to_plot
        }
    }
