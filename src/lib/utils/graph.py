import networkx as nx
import copy as cp
import torch
import re
import torch.nn.functional as tf

from src.lib.utils import misc

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
# 1) current_state (will give output '[current_state]')
# 2) action, current_state (will give output '[action]__[current_state]')
# 3) next_state, action, current_state (will give output '[next_state]__[action]__[current_state]')
def build_stochastic_graph_node(*states_and_actions):
    return '__'.join(['[{}]'.format(state_or_action) for state_or_action in states_and_actions])

def decompose_stochastic_graph_node(stochastic_graph_node):
    return [node.strip('[').strip(']') for node in stochastic_graph_node.split('__')]

# Return True if graph is in stochastic format, False otherwise. This just checks whether
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

def get_unique_single_agent_actions_from_joint_actions(joint_actions):
    action_pairs = [
        multiagent_action_to_single_agent_actions(joint_action) for joint_action in joint_actions
    ]

    return [
        sorted(set([action_pair[0] for action_pair in action_pairs])), # Agent A actions
        sorted(set([action_pair[1] for action_pair in action_pairs])) # Agent B actions
    ]

def get_single_agent_actions_from_joint_mdp_graph(input_graph):
    return get_unique_single_agent_actions_from_joint_actions(get_actions_from_graph(input_graph))

def get_available_actions_from_graph_state(input_graph, state):
    return sorted([
        decompose_stochastic_graph_node(
            action_node
        )[0] for action_node in input_graph.neighbors(build_stochastic_graph_node(state))
    ])

def get_available_states_and_probabilities_from_mdp_graph_state_and_action(input_mdp_graph, state, action):
    action_node = build_stochastic_graph_node(action, state)
    return {
        decompose_stochastic_graph_node(
            next_state_node
        )[0]: input_mdp_graph[action_node][next_state_node]['weight'] for next_state_node in input_mdp_graph.neighbors(
            action_node
        )
    }

def extract_subgraph_containing_states(input_graph, states_to_extract):
    output_graph_ = cp.deepcopy(input_graph)
    output_graph_.remove_nodes_from([
        node for node in output_graph_.nodes if (decompose_stochastic_graph_node(node)[-1] not in states_to_extract)
    ])
    return output_graph_

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
                node_id: 'Reward: {}'.format(
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
            'label': 'gamma = {}'.format(discount_rate_to_plot), 'labelloc': 't'
        }

    return mdp_or_policy_graph_

def graph_to_joint_transition_tensor(mdp_graph, return_sparse=False):
    if is_graph_stochastic(mdp_graph):
        state_list = get_states_from_graph(mdp_graph)
        action_list_A, action_list_B = get_single_agent_actions_from_joint_mdp_graph(mdp_graph)
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
    
# First index is current state s; second index is Agent A action a_A; third index is Agent B action a_B; fourth index is next state s'.
    return misc.to_tensor_representation(transition_tensor_.to(torch.float), to_sparse=return_sparse)

def graph_to_full_transition_tensor(mdp_graph):
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

def graph_to_policy_tensor(policy_graph, return_sparse=False):
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
    
    return misc.to_tensor_representation(policy_tensor_.to(torch.float), to_sparse=return_sparse)

# Relevant calculation: https://drive.google.com/file/d/1XwM_HXkFu1VglsYhsew6SM3BMqb9T_9R/view
def compute_full_transition_tensor(
    joint_transition_tensor,
    policy_tensor,
    acting_agent_is_A=True,
    return_sparse=False
):
    joint_transition_tensor_sparse = misc.sparsify_tensor(joint_transition_tensor)
    policy_tensor_dense = misc.densify_tensor(policy_tensor)

    # Canonical ordering of dimensions is (current state, agent A action, agent B action, next state).
    # If policy_tensor corresponds to the policy for Agent B (and therefore the currently acting agent
    # is A, i.e., acting_agent_is_A == True), transpose axis 1 (agent A action) and axis 2 (agent B action)
    # before proceeding.
    transition_tensor_sparse = torch.transpose(joint_transition_tensor_sparse, 1, 2) if (
        acting_agent_is_A
    ) else joint_transition_tensor_sparse

    return misc.to_tensor_representation(
        torch.sparse.sum(
            transition_tensor_sparse * policy_tensor_dense.unsqueeze(-1).expand(
                -1, -1, transition_tensor_sparse.shape[2]
            ).unsqueeze(-1).expand(
                -1, -1, -1, transition_tensor_sparse.shape[3]
            ).sparse_mask(transition_tensor_sparse.coalesce()),
            dim=1
        ), to_sparse=return_sparse
    )

def graphs_to_full_transition_tensor(joint_mdp_graph, policy_graph, acting_agent_is_A=True, return_sparse=False):
    return compute_full_transition_tensor(
        graph_to_joint_transition_tensor(joint_mdp_graph, return_sparse=True),
        graph_to_policy_tensor(policy_graph, return_sparse=False),
        acting_agent_is_A=acting_agent_is_A,
        return_sparse=return_sparse
    )

def any_graphs_to_full_transition_tensor(*transition_graphs, acting_agent_is_A=True):
    if len(transition_graphs) == 1: # Single agent case
        return graph_to_full_transition_tensor(transition_graphs[0])

    else: # Multiagent case
        return graphs_to_full_transition_tensor(
            *transition_graphs, acting_agent_is_A=acting_agent_is_A
        )

# Calculation details here:
# https://drive.google.com/file/d/1t3wx0P-j3IBKZfD7td9Cewz2-sO5zzDz/view
def compute_state_transition_matrix(full_transition_tensor, policy_tensor):
    return (
        (
            full_transition_tensor
        ) * (
            torch.tile(policy_tensor, (policy_tensor.shape[0], 1, 1)).transpose(0, 1).transpose(1, 2)
        )
    ).sum(dim=1).transpose(0, 1)

def one_step_rollout(state_transition_matrix, state_vector):
    return torch.matmul(state_transition_matrix, state_vector)

def state_to_vector(state, state_list):
    return tf.one_hot(torch.tensor(state_list.index(state)), num_classes=len(state_list)).to(torch.float)
