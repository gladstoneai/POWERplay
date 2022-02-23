import torch
import functools as func
import operator as op
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import copy as cp

################################################################################

TINY_NUMBER = 1e-45 # To avoid both divide by 0 and overflow in torch

################################################################################

def build_sweep_config(
    input_sweep_config=None,
    sweep_name=None,
    project=None,
    entity=None
):
    return {
        **input_sweep_config,
        'parameters': { # Allows us to name individual runs with preset strings
            key: (value_dict if ('value' in input_sweep_config.get('parameters').get(key)) else { 'values': [
                [val, name] for val, name in zip(value_dict.get('values'), value_dict.get('names'))
            ]}) for (key, value_dict) in input_sweep_config.get('parameters').items()
        },
        'name': sweep_name,
        'project': project,
        'entity': entity,
        'method': 'grid'
    }

def retrieve(dictionary, path_with_dots):
    return func.reduce(op.getitem, path_with_dots.split('.'), dictionary)

# Biggest fractional change in state value over one iteration.
def calculate_value_convergence(old_values, new_values, tiny_number=TINY_NUMBER):
    return torch.max(torch.abs((old_values - new_values) / (old_values + tiny_number)))

def pdf_sampler_constructor(pdf=lambda x: 1, interval=(0, 1), resolution=100):

# A bit weird, but lets us match the currying signature of PyTorch distribution functions.
    def constructor_l2():

        def pdf_sampler(number_of_samples):
            source_samples = torch.rand(number_of_samples)

            dest_boundary_points = torch.linspace(*(interval + (resolution,)))
            sample_widths = dest_boundary_points[1:] - dest_boundary_points[:-1]
            pdf_at_midpoints = torch.ones(resolution - 1) * pdf(
                0.5 * (dest_boundary_points[1:] + dest_boundary_points[:-1])
            )
            probability_masses = sample_widths * pdf_at_midpoints / torch.sum(sample_widths * pdf_at_midpoints)

            source_boundary_points = torch.cat((torch.zeros(1), torch.cumsum(probability_masses, 0)))
            sample_bins = torch.bucketize(source_samples, source_boundary_points)
            
            sample_bin_positions = (
                source_samples - source_boundary_points[sample_bins - 1]
            ) / (
                source_boundary_points[sample_bins] - source_boundary_points[sample_bins - 1]
            )

            return sample_bin_positions * (
                dest_boundary_points[sample_bins] - dest_boundary_points[sample_bins - 1]
            ) + dest_boundary_points[sample_bins - 1]
        
        return pdf_sampler
    
    return constructor_l2

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

def build_run_name(local_sweep_name, run_config, sweep_variable_params):
    return '-'.join([local_sweep_name] + [ # sorted() ensures naming is always consistent
        '{0}__{1}'.format(key, run_config[key][1]) for key in sorted(run_config.keys()) if (
            key in sweep_variable_params
        )
    ])

def get_variable_params(sweep_config):
    return [
        param for param in sweep_config.get('parameters').keys() if (
            sweep_config.get('parameters').get(param).get('values') is not None
        )
    ]

def gridworld_state_to_coords(gridworld_state):
    return [int(coord) for coord in str(gridworld_state)[1:-1].split(',')]

def gridworld_states_to_coords(gridworld_state_list):
    return [gridworld_state_to_coords(state) for state in gridworld_state_list]

def gridworld_coords_to_state(gridworld_coords):
    return '({0}, {1})'.format(gridworld_coords[0], gridworld_coords[1])

def gridworld_coords_to_states(gridworld_coords_list):
    return [gridworld_coords_to_state(coords) for coords in gridworld_coords_list]

def transform_graph_for_plots(mdp_graph):
    mdp_graph_ = cp.deepcopy(mdp_graph)

    nx.set_node_attributes(
        mdp_graph_,
        { node_id: str(node_id).split('__')[0] for node_id in list(mdp_graph_) },
        name='label'
    )
    nx.set_node_attributes( # Boxes are states, circles are actions
        mdp_graph_,
        { node_id: (
            'circle' if len(node_id.split('__')) == 2 else 'box'
        ) for node_id in list(mdp_graph_) },
        name='shape'
    )
    nx.set_edge_attributes(
        mdp_graph_,
        nx.get_edge_attributes(mdp_graph_, 'weight'),
        name='label'
    )

    return mdp_graph_

# Return True if graph is in stochastic format, False otherwise. Currently this just checks whether
# some edges in the graph have weights. If none have weights, we conclude the graph is not in
# stochastic format.
def is_graph_stochastic(mdp_graph):
    return (nx.get_edge_attributes(mdp_graph, 'weight') != {})

def get_states_from_graph(graph):
    return [
        node for node in list(graph) if len(node.split('__')) == 1
    ] if is_graph_stochastic(graph) else list(graph)

def get_actions_from_graph(graph):
    return sorted(set([
        node.split('__')[0] for node in list(graph) if len(node.split('__')) == 2
    ])) if is_graph_stochastic(graph) else list(graph)

def graph_to_transition_tensor(graph):
    if is_graph_stochastic(graph):
        state_list, action_list = get_states_from_graph(graph), get_actions_from_graph(graph)
        transition_tensor_ = torch.zeros(len(state_list), len(action_list), len(state_list))

        for i in range(len(state_list)):
            for j in range(len(action_list)):
                for k in range(len(state_list)):

                    try:
                        transition_tensor_[i][j][k] = graph['__'.join(
                            [action_list[j], state_list[i]]
                        )]['__'.join(
                            [state_list[k], action_list[j], state_list[i]]
                        )]['weight']
# Some state-action-state triples don't occur; transition_tensor_ entry remains zero in those cases.
                    except KeyError:
                        pass
    
    else:
        transition_tensor_ = torch.diag_embed(torch.tensor(nx.to_numpy_array(graph)))
    
    return transition_tensor_.to(torch.float)

def clone_run_inputs(runs_data_with_transition_tensor, ignore_terminal_state=False):
    return {
        run_data['name']: {
            'args': [
                run_data['outputs']['reward_samples'][:,:-1] if (
                    ignore_terminal_state
                ) else run_data['outputs']['reward_samples'],
                run_data['inputs']['transition_tensor'],
                run_data['inputs']['discount_rate']
            ], 'kwargs': {
                'num_workers': run_data['inputs']['num_workers'],
                'convergence_threshold': run_data['inputs']['convergence_threshold']
            }
         } for run_data in runs_data_with_transition_tensor.values()
    }
