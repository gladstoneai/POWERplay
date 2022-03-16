import networkx as nx
import copy as cp
import torch

def gridworld_state_to_coords(gridworld_state):
    return [int(coord) for coord in str(gridworld_state)[1:-1].split(',')]

def gridworld_states_to_coords(gridworld_state_list):
    return [gridworld_state_to_coords(state) for state in gridworld_state_list]

def gridworld_coords_to_state(gridworld_coords):
    return '({0}, {1})'.format(gridworld_coords[0], gridworld_coords[1])

def gridworld_coords_to_states(gridworld_coords_list):
    return [gridworld_coords_to_state(coords) for coords in gridworld_coords_list]

# Builds a correctly formatted node in the stochastic graph. Input should be either:
# 1) state1
# 2) state1, action
# 3) state1, action, state2
# Output format is '[state_1]__[action]__[state2]'
def build_stochastic_graph_node(*states_and_actions):
    return '__'.join(['[{}]'.format(state_or_action) for state_or_action in states_and_actions])

def decompose_stochastic_graph_node(stochastic_graph_node):
    return [node.strip('[').strip(']') for node in stochastic_graph_node.split('__')]

def transform_graph_for_plots(mdp_or_policy_graph):
    mdp_or_policy_graph_ = cp.deepcopy(mdp_or_policy_graph)

    nx.set_node_attributes(
        mdp_or_policy_graph_,
        { node_id: decompose_stochastic_graph_node(node_id)[0] for node_id in list(mdp_or_policy_graph_) },
        name='label'
    )
    nx.set_node_attributes( # Boxes are states, circles are actions
        mdp_or_policy_graph_,
        { node_id: (
            'circle' if len(decompose_stochastic_graph_node(node_id)) == 2 else 'box'
        ) for node_id in list(mdp_or_policy_graph_) },
        name='shape'
    )
    nx.set_edge_attributes(
        mdp_or_policy_graph_,
        {
            edge: round(weight, 3) for edge, weight in nx.get_edge_attributes(
                mdp_or_policy_graph_, 'weight'
            ).items()
        },
        name='label'
    )

    return mdp_or_policy_graph_

# Return True if graph is in stochastic format, False otherwise. Currently this just checks whether
# some edges in the graph have weights. If none have weights, we conclude the graph is not in
# stochastic format.
def is_graph_stochastic(mdp_graph):
    return (nx.get_edge_attributes(mdp_graph, 'weight') != {})

def graph_to_transition_tensor(graph):
    if is_graph_stochastic(graph):
        state_list, action_list = get_states_from_graph(graph), get_actions_from_graph(graph)
        transition_tensor_ = torch.zeros(len(state_list), len(action_list), len(state_list))

        for i in range(len(state_list)):
            for j in range(len(action_list)):
                for k in range(len(state_list)):

                    try:
                        transition_tensor_[i][j][k] = graph[
                            build_stochastic_graph_node(action_list[j], state_list[i])
                        ][
                            build_stochastic_graph_node(state_list[k], action_list[j], state_list[i])
                        ]['weight']
# Some state-action-state triples don't occur; transition_tensor_ entry remains zero in those cases.
                    except KeyError:
                        pass
    
    else:
        transition_tensor_ = torch.diag_embed(torch.tensor(nx.to_numpy_array(graph)))

# First index is current state s; second index is action a; third index is next state s'.
    return transition_tensor_.to(torch.float)

def get_states_from_graph(graph):
    return [
        decompose_stochastic_graph_node(node)[0] for node in list(graph) if (
            len(decompose_stochastic_graph_node(node)) == 1
        )
    ] if is_graph_stochastic(graph) else list(graph)

def get_actions_from_graph(graph):
    return sorted(set([
        decompose_stochastic_graph_node(node)[0] for node in list(graph) if (
            len(decompose_stochastic_graph_node(node)) == 2
        )
    ])) if is_graph_stochastic(graph) else list(graph)