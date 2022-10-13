import matplotlib.pyplot as plt
import collections as col
import math

from .lib.utils import graph
from .lib.utils import render
from .lib import data
from .lib import check

def plot_sample_aggregations(
    all_samples,
    state_list,
    show=True,
    plot_as_gridworld=False,
    save_handle=None,
    aggregation='mean',
    sample_quantity='POWER',
    sample_units='reward units',
    manual_title=None,
    save_figure=data.save_figure,
    save_folder=data.EXPERIMENT_FOLDER
):
    if plot_as_gridworld:
        fig = render.render_gridworld_aggregations(
            all_samples,
            state_list,
            aggregation=aggregation,
            sample_quantity=sample_quantity,
            sample_units=sample_units,
            manual_title=manual_title
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
    plot_in_log_scale=False,
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
        sample_units=sample_units,
        plot_in_log_scale=plot_in_log_scale
    )

    if callable(save_figure) and save_handle is not None:
        for fig in all_figs:
            save_figure(fig, '{0}_samples{1}-{2}'.format(
                sample_quantity,
                '_agent_A_state_{}'.format(fig.agent_A_state) if (
                    hasattr(fig, 'agent_A_state')
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
                    '_agent_A_state{}'.format(fig.agent_A_state) if (
                        hasattr(fig, 'agent_A_state')
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
    number_of_states_per_figure=128,
    reward_to_plot=None,
    discount_rate_to_plot=None,
    save_handle='temp',
    graph_name='mdp_graph',
    save_folder=data.TEMP_FOLDER,
    temp_folder=data.TEMP_FOLDER
):
    state_list = graph.get_states_from_graph(mdp_or_policy_graph)
    chunked_state_list = [
        state_list[i:number_of_states_per_figure + i] for i in range(
            0, len(state_list), number_of_states_per_figure
        )
    ]
    all_figs_ = []

    for i in range(len(chunked_state_list)):
        fig_number_label = str(i)

        graph_to_plot = graph.transform_graph_for_plots(
            graph.extract_subgraph_containing_states(mdp_or_policy_graph, chunked_state_list[i]),
            reward_to_plot=reward_to_plot,
            discount_rate_to_plot=discount_rate_to_plot
        )
        
    # We save to temp solely for the purpose of plotting, since Graphviz prefers to consume files
    # rather than literal dot strings. We save it in temp so as not to overwrite "permanent" MDP graphs
    # and so git doesn't track the contents of the temp folder.
        data.save_graph_to_dot_file(graph_to_plot, save_handle, folder=temp_folder)

        fig_ = data.create_and_save_mdp_figure(
            save_handle,
            subgraphs_per_row=subgraphs_per_row,
            fig_name='{0}_{1}-{2}'.format(graph_name, fig_number_label, save_handle),
            mdps_folder=temp_folder,
            fig_folder=save_folder,
            show=show
        )
        fig_.number_label = fig_number_label

        all_figs_ += [fig_]

    return all_figs_

def plot_all_outputs(
    reward_samples,
    power_samples,
    state_list,
    graphs_to_plot=[],
    plot_distributions=True,
    plot_correlations=True,
    plot_in_log_scale=False,
    **kwargs
):
    max_nodes_per_figure = 1024 # For MDP graphs bigger than this, we chunk the plot into multiple figures

    for graph_to_plot in graphs_to_plot:
        check.check_state_list_identity(state_list, graph.get_states_from_graph(graph_to_plot['graph_data']))

    mdp_kwargs = {
        key: kwargs[key] for key in list(
            set(kwargs.keys()) & set(['show', 'save_handle', 'save_folder', 'temp_folder'])
        )
    }

    print()
    print('Rendering plots...')

    return {
        'POWER means': plot_sample_aggregations(
            power_samples, state_list, aggregation='mean', sample_quantity='POWER', sample_units='reward units', **kwargs
        ),
        'POWER variances': plot_sample_aggregations(
            power_samples, state_list, aggregation='var', sample_quantity='POWER', sample_units='reward units', **kwargs
        ),
        **({
            'Reward samples over states{}'.format(
                ', Agent A at {}'.format(fig.agent_A_state) if hasattr(fig, 'agent_A_state') else ''
            ): fig for fig in plot_sample_distributions(
                reward_samples,
                state_list,
                sample_quantity='Reward',
                sample_units='reward units',
                plot_in_log_scale=plot_in_log_scale,
                **kwargs
            )
        } if plot_distributions else {}),
        **({
            'POWER samples over states{}'.format(
                ', Agent A at {}'.format(fig.agent_A_state) if hasattr(fig, 'agent_A_state') else ''
            ): fig for fig in plot_sample_distributions(
                reward_samples,
                state_list,
                sample_quantity='POWER',
                sample_units='reward units',
                plot_in_log_scale=plot_in_log_scale,
                **kwargs
            )
        } if plot_distributions else {}),
        **(
            dict(col.ChainMap(*[{
                'POWER correlations, state {0}{1}'.format(
                    state, ', Agent A at {}'.format(fig.agent_A_state) if hasattr(fig, 'agent_A_state') else ''
                ): fig for fig in plot_sample_correlations(
                    power_samples, state_list, state, sample_quantity='POWER', sample_units='reward units', **kwargs
                )
            } for state in state_list])) if plot_correlations else {}
        ),
         **(
            dict(col.ChainMap(*[{
                '{0}, Figure {1}'.format(
                    graph_to_plot['graph_description'], fig.number_label
                ): fig for fig in plot_mdp_or_policy(
                    graph_to_plot['graph_data'], **{
                        **mdp_kwargs,
                        'graph_name': graph_to_plot['graph_name'],
                        'number_of_states_per_figure': math.floor(
                            (
                                max_nodes_per_figure / len(list(graph_to_plot['graph_data']))
                            ) * len(graph.get_states_from_graph(graph_to_plot['graph_data']))
                        )
                    }
                )
            } for graph_to_plot in graphs_to_plot]))
        )
    }
