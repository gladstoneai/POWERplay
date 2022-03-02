import networkx as nx
import copy as cp
import itertools as it
import torch
import numpy as np

from . import utils
from . import viz
from . import check

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

    row_coord_lims, col_coord_lims = np.array(utils.gridworld_states_to_coords([corner1_state, corner2_state])).T
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

def add_state_action(mdp_graph, state_to_add, action_dict, check_closure=False):
    check.check_name_for_underscores(state_to_add)
    check.check_action_dict(action_dict)

    mdp_graph_ = cp.deepcopy(mdp_graph)

    mdp_graph_.add_node(state_to_add)

    for action in action_dict.keys():
        action_node_id = '__'.join([action, state_to_add])
        mdp_graph_.add_node(action_node_id)
        mdp_graph_.add_edge(state_to_add, action_node_id)

        for state in action_dict[action].keys():
            state_node_id = '__'.join([state, action, state_to_add])
            mdp_graph_.add_node(state_node_id)
            mdp_graph_.add_edge(action_node_id, state_node_id, weight=action_dict[action][state])
    
    if check_closure:
        check.check_stochastic_mdp_closure(mdp_graph_)

    return mdp_graph_

def remove_state_action(mdp_graph, state_to_remove, check_closure=False):
    mdp_graph_ = cp.deepcopy(mdp_graph)
    mdp_graph_.remove_nodes_from([
        stoch_state for stoch_state in mdp_graph_ if stoch_state.split('__')[-1] == state_to_remove
    ])

    if check_closure:
        check.check_stochastic_mdp_closure(mdp_graph_)
    
    return mdp_graph_

def update_state_action(mdp_graph, state_to_update, new_action_dict, check_closure=False):
    mdp_graph_ = cp.deepcopy(mdp_graph)
    mdp_graph_ = remove_state_action(mdp_graph_, state_to_update, check_closure=False)

    return add_state_action(mdp_graph_, state_to_update, new_action_dict, check_closure=check_closure)

def mdp_to_stochastic_graph(mdp_graph):
    stochastic_graph_ = nx.DiGraph()

    for state in utils.get_states_from_graph(mdp_graph):
        stochastic_graph_ = add_state_action(
            stochastic_graph_,
            state,
            { next_state: { next_state: 1 } for next_state in mdp_graph.successors(state) }
        )
    
    return stochastic_graph_

def gridworld_to_stochastic_graph(
    gridworld_mdp,
    stochastic_noise_level=0,
    noise_bias={}
):
    check.check_stochastic_noise_level(stochastic_noise_level)
    check.check_noise_bias(noise_bias, stochastic_noise_level)

    stochastic_graph_ = nx.DiGraph()
    grid_coords_list = utils.gridworld_states_to_coords(utils.get_states_from_graph(gridworld_mdp))

    possible_transitions = {
        'up': lambda coords: [coords[0] - 1, coords[1]] if (
            [coords[0] - 1, coords[1]] in grid_coords_list
        ) else None,
        'down': lambda coords: [coords[0] + 1, coords[1]] if (
            [coords[0] + 1, coords[1]] in grid_coords_list
        ) else None,
        'left': lambda coords: [coords[0], coords[1] - 1] if (
            [coords[0], coords[1] - 1] in grid_coords_list
        ) else None,
        'right': lambda coords: [coords[0], coords[1] + 1] if (
            [coords[0], coords[1] + 1] in grid_coords_list
        ) else None,
        'stay': lambda coords: coords
    }

    default_transition = 'stay'

    for grid_coords in grid_coords_list:
        allowed_actions = [action for action, transition in possible_transitions.items() if (
            transition(grid_coords) is not None
        )]

        allowed_noise_bias_ = {}
        # If the bias direction is blocked, stack the bias probability from that direction onto the
        # default_transition direction. So if the bias goes 'left' 0.2 and 'left' is blocked, stack an
        # extra 0.2 probability on 'stay'.
        for noise_bias_direction, noise_bias_amount in noise_bias.items():
            if noise_bias_direction in allowed_actions:
                allowed_noise_bias_[noise_bias_direction] = noise_bias_amount
            elif allowed_noise_bias_.get(default_transition) is not None:
                allowed_noise_bias_[default_transition] += noise_bias_amount
            else:
                allowed_noise_bias_[default_transition] = noise_bias_amount

        unbiased_noise = (
            stochastic_noise_level - sum([noise for noise in allowed_noise_bias_.values()])
        ) / (len(allowed_actions) - len(allowed_noise_bias_))

        stochastic_graph_ = add_state_action(
            stochastic_graph_,
            utils.gridworld_coords_to_state(grid_coords),
            {
                action: { # These are transition probabilities for states s' that do *not* correspond to the action taken
                    **{
                        utils.gridworld_coords_to_state(
                            possible_transitions[allowed_action](grid_coords)
                        ): (
                            allowed_noise_bias_.get(allowed_action) if (
                                allowed_action in allowed_noise_bias_.keys()
                            ) else unbiased_noise
                        ) for allowed_action in allowed_actions if stochastic_noise_level > 0
                    },
                    **{ # This is the transition probability for the state s' that *does* correspond to the action taken
                        utils.gridworld_coords_to_state(
                            possible_transitions[action](grid_coords)
                        ): (
                            1 - stochastic_noise_level + allowed_noise_bias_.get(action) if (
                                action in allowed_noise_bias_.keys()
                            ) else 1 - stochastic_noise_level + unbiased_noise
                        )
                    }
                } for action in allowed_actions
            }
        )
    
    return stochastic_graph_

def generate_noised_gridworlds(gridworld_mdp, noise_level_list=[], noise_bias_list=[]):
    return [
        gridworld_to_stochastic_graph(
            gridworld_mdp, stochastic_noise_level=noise_level, noise_bias=noise_bias
        ) for noise_level, noise_bias in zip(noise_level_list, noise_bias_list)
    ]

def view_gridworld(gridworld_mdp):
    viz.plot_sample_aggregations(
        torch.zeros((1, len(list(gridworld_mdp)))),
        [str(state) for state in gridworld_mdp],
        plot_as_gridworld=True,
        sample_quantity='',
        sample_units='dummy data'
    )