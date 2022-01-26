import networkx as nx
import copy as cp
import itertools as it
import torch

from . import utils
from . import viz

# Converts an undirected graph into a digraph and adds self-loops to all "absorbing" states.
# Basically, a quick and dirty way to convert default NetworkX graphs into graphs compatible with our MDP
# conventions.
def quick_graph_to_mdp(mdp_graph, name=''):
    return nx.DiGraph(
        list(mdp_graph.edges) + [
            str((node, node)) for node in mdp_graph.nodes() if node not in [edge[0] for edge in mdp_graph.edges()]
        ], name=name
    )

# Adds self-loops to all states in a digraph.
def mdp_add_self_loops(mdp_graph):
    return nx.DiGraph(list(nx.DiGraph(mdp_graph).edges) + [(node, node) for node in mdp_graph.nodes()], name=mdp_graph.name)

# Deletes all states of a gridworld in the square defined by corner1_state and corner2_state.
# e.g., corner1_state='(0, 0)', corner1_state='(2, 2)' deletes all states in the square:
# '(0, 0)', '(0, 1)', '(0, 2)', '(1, 0)', '(1, 1)', '(1, 2)', '(2, 0)', '(2, 1)', '(2, 2)'
def delete_gridworld_square(gridworld_mdp_graph, corner1_state, corner2_state):
    gridworld_mdp_graph_ = cp.deepcopy(gridworld_mdp_graph)

    row_coord_lims, col_coord_lims = utils.gridworld_coords_from_states([corner1_state, corner2_state])
    rows_to_delete = list(range(min(row_coord_lims), max(row_coord_lims) + 1))
    cols_to_delete = list(range(min(col_coord_lims), max(col_coord_lims) + 1))

    states_to_delete = [state for state in it.product(rows_to_delete, cols_to_delete)]
    # The state may be saved as a str or a tuple, so we delete both kinds.
    gridworld_mdp_graph_.remove_nodes_from(states_to_delete + [str(state) for state in states_to_delete])

    return gridworld_mdp_graph_

def construct_gridworld(num_rows, num_cols, name='custom gridworld', squares_to_delete=[]):
    gridworld_mdp_ = quick_graph_to_mdp(
        nx.generators.lattice.grid_2d_graph(num_rows, num_cols, create_using=nx.DiGraph()), name=name
    )
    
    for square_corners in squares_to_delete:
        gridworld_mdp_ = delete_gridworld_square(gridworld_mdp_, square_corners[0], square_corners[1])
    
    return mdp_add_self_loops(gridworld_mdp_)

def view_gridworld(gridworld_mdp):
    viz.plot_sample_means(
        torch.zeros((1, len(list(gridworld_mdp)))),
        [str(state) for state in gridworld_mdp],
        plot_as_gridworld=True,
        sample_quantity='',
        sample_units='dummy data'
    )