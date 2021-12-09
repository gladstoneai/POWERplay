import torch
import functools as func
import operator as op
import math
import matplotlib.pyplot as plt
import networkx as nx

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

def generate_fig_layout(number_of_subplots, sharey=True):
    fig_cols = min(number_of_subplots, 4)
    fig_rows = math.ceil(number_of_subplots / fig_cols)

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

def graph_to_adjacency_matrix(mdp_graph, state_list=None):
    return torch.tensor(nx.to_numpy_array(mdp_graph, nodelist=state_list))

# Converts an undirected graph into a digraph; adds self-loops to all "absorbing" states; adds a TERMINAL state.
# Basically, a quick and dirty way to convert default NetworkX graphs into graphs compatible with our MDP
# conventions.
def quick_graph_to_mdp(mdp_graph):
    return nx.DiGraph(
        list(mdp_graph.edges) + [
            (node, node) for node in mdp_graph.nodes() if node not in [edge[0] for edge in mdp_graph.edges()]
        ] + [('TERMINAL', 'TERMINAL')]
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