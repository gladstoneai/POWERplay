import networkx as nx
import copy as cp
import torch
import re
import torch.nn.functional as tf

def gridworld_state_to_coords(gridworld_state):
    return [int(coord) for coord in str(gridworld_state)[1:-1].split(',')]

def gridworld_states_to_coords(gridworld_state_list):
    return [gridworld_state_to_coords(state) for state in gridworld_state_list]

def gridworld_coords_to_state(gridworld_row_coord, gridworld_col_coord):
    return '({0}, {1})'.format(gridworld_row_coord, gridworld_col_coord)

def gridworld_coords_to_states(gridworld_coords_list):
    return [gridworld_coords_to_state(*coords) for coords in gridworld_coords_list]

def single_agent_states_to_multiagent_state(agent_A_state, agent_B_state):
    return '{0}_A^{1}_B'.format(agent_A_state, agent_B_state)

def single_agent_actions_to_multiagent_action(agent_A_action, agent_B_action):
    return single_agent_states_to_multiagent_state(agent_A_action, agent_B_action)

def multiagent_state_to_single_agent_states(multiagent_state):
    return [state.split('_')[0] for state in multiagent_state.split('^')]

def multiagent_action_to_single_agent_actions(multiagent_action):
    return multiagent_state_to_single_agent_states(multiagent_action)

def multiagent_states_to_single_agent_states(multiagent_state_list):
    return [
        multiagent_state_to_single_agent_states(
            multiagent_state
        ) for multiagent_state in multiagent_state_list
    ]

# Builds a correctly formatted node in the stochastic graph. Input should be either:
# 1) state1
# 2) state1, action
# 3) state1, action, state2
# Output format is '[state_1]__[action]__[state2]'
def build_stochastic_graph_node(*states_and_actions):
    return '__'.join(['[{}]'.format(state_or_action) for state_or_action in states_and_actions])

def decompose_stochastic_graph_node(stochastic_graph_node):
    return [node.strip('[').strip(']') for node in stochastic_graph_node.split('__')]

# Return True if graph is in stochastic format, False otherwise. Currently this just checks whether
# some edges in the graph have weights. If none have weights, we conclude the graph is not in
# stochastic format.
def is_graph_stochastic(input_graph):
    return (nx.get_edge_attributes(input_graph, 'weight') != {})

def get_states_from_graph(input_graph):
    return sorted([
        decompose_stochastic_graph_node(node)[0] for node in list(input_graph) if (
            len(decompose_stochastic_graph_node(node)) == 1
        )
    ]) if is_graph_stochastic(input_graph) else list(input_graph)

def get_actions_from_graph(input_graph):
    return sorted(set([
        decompose_stochastic_graph_node(node)[0] for node in list(input_graph) if (
            len(decompose_stochastic_graph_node(node)) == 2
        )
    ])) if is_graph_stochastic(input_graph) else list(input_graph)

def get_single_agent_actions_from_multiagent_graph(input_graph):
    action_pairs = [
        multiagent_action_to_single_agent_actions(joint_action) for joint_action in get_actions_from_graph(input_graph)
    ]

    return [
        sorted(set([action_pair[0] for action_pair in action_pairs])),
        sorted(set([action_pair[1] for action_pair in action_pairs]))
    ]

def get_available_actions_from_graph_state(input_graph, state):
    return [
        decompose_stochastic_graph_node(
            action_node
        )[0] for action_node in input_graph.neighbors(build_stochastic_graph_node(state))
    ]

def get_available_states_and_probabilities_from_mdp_graph_state_and_action(input_mdp_graph, state, action):
    action_node = build_stochastic_graph_node(action, state)
    return {
        decompose_stochastic_graph_node(
            next_state_node
        )[0]: input_mdp_graph[action_node][next_state_node]['weight'] for next_state_node in input_mdp_graph.neighbors(
            action_node
        )
    }

def are_gridworld_states_multiagent(state_list):
    return all([
        bool(re.fullmatch(r'\(\d+, \d+\)_A\^\(\d+, \d+\)_B', state)) for state in state_list
    ])

def are_general_graph_states_multiagent(state_list):
    return all([
        bool(re.fullmatch(r'.*_A\^.*_B', state)) for state in state_list
    ])

def is_graph_multiagent(input_graph):
    return are_general_graph_states_multiagent(get_states_from_graph(input_graph))

def transform_graph_for_plots(mdp_or_policy_graph, reward_to_plot=None, discount_rate_to_plot=None):
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

    if reward_to_plot is not None:
        nx.set_node_attributes(
            mdp_or_policy_graph_,
            {
                node_id: 'Reward: {0}'.format(
                    round(float(reward_to_plot[
                        get_states_from_graph(mdp_or_policy_graph_).index(
                            decompose_stochastic_graph_node(node_id)[0]
                        )
                    ]), 4)
                ) for node_id in list(mdp_or_policy_graph_) if (
                    len(decompose_stochastic_graph_node(node_id)) == 1
                )
            },
            name='xlabel'
        )
    
    if discount_rate_to_plot is not None:
        mdp_or_policy_graph_.graph['graph'] = {
            'label': 'gamma = {0}'.format(discount_rate_to_plot), 'labelloc': 't'
        }

    return mdp_or_policy_graph_

def graph_to_simultaneous_transition_tensor(mdp_graph):
    if is_graph_stochastic(mdp_graph):
        state_list = get_states_from_graph(mdp_graph)
        action_list_A, action_list_B = get_single_agent_actions_from_multiagent_graph(mdp_graph)
        transition_tensor_ = torch.zeros(len(state_list), len(action_list_A), len(action_list_B), len(state_list))

        for i in range(len(state_list)):
            for j in range(len(action_list_A)):
                for k in range(len(action_list_B)):
                    for l in range(len(state_list)):

                        try:
                            transition_tensor_[i][j][k][l] = mdp_graph[
                                build_stochastic_graph_node(
                                    single_agent_actions_to_multiagent_action(action_list_A[j], action_list_B[k]),
                                    state_list[i]
                                )
                            ][
                                build_stochastic_graph_node(
                                    state_list[l],
                                    single_agent_actions_to_multiagent_action(action_list_A[j], action_list_B[k]),
                                    state_list[i]
                                )
                            ]['weight']

# Some state-action-action-state quadruples don't occur; transition_tensor_ entry remains zero in those cases.
                        except KeyError:
                            pass
    
    else:
        transition_tensor_ = torch.diag_embed(torch.tensor(nx.to_numpy_array(mdp_graph)))
    
    return transition_tensor_

def graph_to_transition_tensor(mdp_graph):
    if is_graph_stochastic(mdp_graph):
        state_list, action_list = get_states_from_graph(mdp_graph), get_actions_from_graph(mdp_graph)
        transition_tensor_ = torch.zeros(len(state_list), len(action_list), len(state_list))

        for i in range(len(state_list)):
            for j in range(len(action_list)):
                for k in range(len(state_list)):

                    try:
                        transition_tensor_[i][j][k] = mdp_graph[
                            build_stochastic_graph_node(action_list[j], state_list[i])
                        ][
                            build_stochastic_graph_node(state_list[k], action_list[j], state_list[i])
                        ]['weight']

# Some state-action-state triples don't occur; transition_tensor_ entry remains zero in those cases.
                    except KeyError:
                        pass
    
    else:
        transition_tensor_ = torch.diag_embed(torch.tensor(nx.to_numpy_array(mdp_graph)))

# First index is current state s; second index is action a; third index is next state s'.
    return transition_tensor_.to(torch.float)

def graph_to_policy_tensor(policy_graph):
    state_list, action_list = get_states_from_graph(policy_graph), get_actions_from_graph(policy_graph)
    policy_tensor_ = torch.zeros(len(state_list), len(action_list))

    for i in range(len(state_list)):
        for j in range(len(action_list)):

            try:
                policy_tensor_[i][j] = policy_graph[
                    build_stochastic_graph_node(state_list[i])
                ][
                    build_stochastic_graph_node(action_list[j], state_list[i])
                ]['weight']

# Some state-action pairs don't occur; policy_tensor_ entry remains zero in those cases.
            except KeyError:
                pass
    
    return policy_tensor_.to(torch.float)

def compute_multiagent_transition_tensor(transition_tensor_A, policy_tensor_B, transition_tensor_B):
    num_states = transition_tensor_A.shape[0]
    num_actions_A = transition_tensor_A.shape[1]
    num_actions_B = transition_tensor_B.shape[1]

    agent_B_state_mapping = (transition_tensor_B * policy_tensor_B.unsqueeze(-1).expand(
        num_states, num_actions_B, num_states
    )).sum(dim=1)

    return ((
        agent_B_state_mapping.unsqueeze(0).unsqueeze(0).expand(
            num_states, num_actions_A, num_states, num_states
        )
    ) * (
        transition_tensor_A.unsqueeze(-1).expand(
            num_states, num_actions_A, num_states, num_states
        )
    )).sum(dim=2)

def graphs_to_multiagent_transition_tensor(mdp_graph_A, policy_graph_B, mdp_graph_B):
    return compute_multiagent_transition_tensor(
        graph_to_transition_tensor(mdp_graph_A),
        graph_to_policy_tensor(policy_graph_B),
        graph_to_transition_tensor(mdp_graph_B)
    )
    
def any_graphs_to_transition_tensor(*transition_graphs):
    if len(transition_graphs) == 1: # Single agent case
        return graph_to_transition_tensor(transition_graphs[0])

    else: # Multiagent case
        return graphs_to_multiagent_transition_tensor(*transition_graphs)

# Calculation details here:
# https://drive.google.com/file/d/1t3wx0P-j3IBKZfD7td9Cewz2-sO5zzDz/view
def one_step_rollout(state_vector, policy_tensor, transition_tensor):
    return ( 
        transition_tensor
        * torch.tile(policy_tensor, (policy_tensor.shape[0], 1, 1)).transpose(0, 1).transpose(1, 2)
        * torch.tile(state_vector, (policy_tensor.shape[0], policy_tensor.shape[1], 1)).transpose(0, 2)
    ).sum(dim=[0, 1])

def state_to_vector(state, state_list):
    return tf.one_hot(torch.tensor(state_list.index(state)), num_classes=len(state_list))
