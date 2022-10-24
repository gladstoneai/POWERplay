import copy as cp
import networkx as nx

from .lib.utils import graph
from .lib.utils import dist
from .lib.utils import misc
from .lib.utils import learn
from .lib import check
from . import multi
from . import mdp

def single_agent_to_multiagent_policy_graph(single_agent_policy_graph, acting_agent_is_H=True):
    multiagent_graph_ = nx.DiGraph()

    for other_agent_state in graph.get_states_from_graph(single_agent_policy_graph):
        multiagent_graph_ = nx.compose(
            multiagent_graph_,
            nx.relabel_nodes(cp.deepcopy(single_agent_policy_graph), {
                node: multi.single_agent_to_multiagent_graph_node(
                    node, other_agent_state, acting_agent_is_H=acting_agent_is_H
                ) for node in single_agent_policy_graph.nodes
            })
        )
    
    return multiagent_graph_

def create_single_agent_random_policy(mdp_graph):
    policy_graph_ = cp.deepcopy(mdp_graph)

    for state in graph.get_states_from_graph(policy_graph_):
        state_edges = policy_graph_.edges(graph.build_stochastic_graph_node(state))

        nx.set_edge_attributes(
            policy_graph_,
            { edge: (1 / len(state_edges)) for edge in state_edges },
            name='weight'
        )
    
    policy_graph_.remove_nodes_from([
        node for node in policy_graph_.nodes if len(graph.decompose_stochastic_graph_node(node)) == 3
    ])

    return policy_graph_

def create_joint_multiagent_random_policy(mdp_graph, acting_agent_is_H=True):
    policy_graph_ = nx.DiGraph()

    for state in graph.get_states_from_graph(mdp_graph):
        action_pairs = [
            graph.multiagent_action_to_single_agent_actions(
                joint_action
            ) for joint_action in graph.get_available_actions_from_graph_state(mdp_graph, state)
        ]

        unique_actions = list(set(
            [
                action_pair[0] for action_pair in action_pairs
            ] if acting_agent_is_H else [
                action_pair[1] for action_pair in action_pairs
            ]
        ))

        policy_graph_ = mdp.add_state_action(policy_graph_, state, {
            action: { 'TEMP': 1 } for action in unique_actions # This next-state node will be immediately deleted
        })

        policy_graph_.remove_nodes_from([
            node for node in policy_graph_.nodes if len(graph.decompose_stochastic_graph_node(node)) == 3
        ])

        state_edges = policy_graph_.edges(graph.build_stochastic_graph_node(state))

        nx.set_edge_attributes(
            policy_graph_,
            { edge: (1 / len(state_edges)) for edge in state_edges },
            name='weight'
        )
    
    return policy_graph_

# acting_agent_is_H=True => This policy is for Agent H.
# acting_agent_is_H=False => This policy is for Agent A.
# Typically we make seed policies for Agent A, so the default is False.
def quick_mdp_to_policy(mdp_graph, acting_agent_is_H=False):
    if graph.is_graph_multiagent(mdp_graph):
        return create_joint_multiagent_random_policy(mdp_graph, acting_agent_is_H=acting_agent_is_H)

    else:
        return create_single_agent_random_policy(mdp_graph)

def update_state_actions(policy_graph, state, new_policy_actions):
    check.check_state_in_graph_states(policy_graph, state)
    check.check_policy_actions(policy_graph, state, new_policy_actions)

    policy_graph_ = cp.deepcopy(policy_graph)

    nx.set_edge_attributes(
        policy_graph_,
        {
            edge: new_policy_actions[
                graph.decompose_stochastic_graph_node(edge[1])[0]
            ] for edge in policy_graph_.edges(graph.build_stochastic_graph_node(state))
        },
        name='weight'
    )

    return policy_graph_

def single_agent_policy_tensor_to_graph(policy_tensor, associated_mdp_graph):
    policy_graph_ = quick_mdp_to_policy(associated_mdp_graph)
    state_list = graph.get_states_from_graph(associated_mdp_graph)
    action_list = graph.get_actions_from_graph(associated_mdp_graph)

    for state in state_list:
        policy_graph_ = update_state_actions(policy_graph_, state, {
            action: float(
                policy_tensor[state_list.index(state)][action_list.index(action)]
            ) for action in graph.get_available_actions_from_graph_state(associated_mdp_graph, state)
        })

    return policy_graph_

# acting_agent_is_H == True means that the policy_tensor belongs to Agent H
def multiagent_policy_tensor_to_graph(policy_tensor, associated_mdp_graph, acting_agent_is_H=True):
    policy_graph_ = quick_mdp_to_policy(associated_mdp_graph, acting_agent_is_H=acting_agent_is_H)
    state_list = graph.get_states_from_graph(associated_mdp_graph)
    action_list_H, action_list_A = graph.get_single_agent_actions_from_joint_mdp_graph(associated_mdp_graph)
    action_list = action_list_H if acting_agent_is_H else action_list_A

    for state in state_list:
        available_actions_H, available_actions_A = graph.get_unique_single_agent_actions_from_joint_actions(
            graph.get_available_actions_from_graph_state(associated_mdp_graph, state)
        )
        available_actions = available_actions_H if acting_agent_is_H else available_actions_A

        policy_graph_ = update_state_actions(policy_graph_, state, {
            action: float(
                policy_tensor[state_list.index(state)][action_list.index(action)]
            ) for action in available_actions
        })
    
    return policy_graph_

# NOTE: state_action_graph must have states and actions that correspond to those of policy_tensor.
# acting_agent_is_H == True means that the policy_tensor belongs to Agent H
def policy_tensor_to_graph(policy_tensor, associated_mdp_graph, acting_agent_is_H=True):
    if graph.is_graph_multiagent(associated_mdp_graph):
        return multiagent_policy_tensor_to_graph(
            policy_tensor, associated_mdp_graph, acting_agent_is_H=acting_agent_is_H
        )
    
    else:
        return single_agent_policy_tensor_to_graph(policy_tensor, associated_mdp_graph)

def sample_optimal_policy_data_from_run(run_properties, reward_sample_index=0):

    reward_function_H, discount_rate_H, transition_graphs, convergence_threshold, sweep_type = (
        run_properties['reward_samples'][reward_sample_index],
        run_properties['discount_rate'],
        run_properties['transition_graphs'],
        run_properties['convergence_threshold'],
        run_properties['sweep_type']
    )

    policy_graph_H = policy_tensor_to_graph(
        learn.find_optimal_policy(
            reward_function_H,
            discount_rate_H,
            graph.any_graphs_to_full_transition_tensor(
                *transition_graphs, acting_agent_is_H=True, return_sparse=False
            ),
            value_initialization=None,
            convergence_threshold=convergence_threshold
        ), # This works for the agent of a single-agent system, OR for Agent H of a multi-agent system
        transition_graphs[0],
        acting_agent_is_H=True
    )

    if sweep_type == 'single_agent':
        return {
            'mdp_graph': transition_graphs[0],
            'policy_graph_H': policy_graph_H,
            'reward_function_H': reward_function_H
        }
    
    elif sweep_type == 'multiagent_fixed_policy':
        return {
            'mdp_graph': transition_graphs[0],
            'policy_graph_H': policy_graph_H,
            'reward_function_H': reward_function_H,
            'policy_graph_A': transition_graphs[1]
        }

    elif sweep_type == 'multiagent_with_reward':
        reward_function_A, discount_rate_A = (
            run_properties['reward_samples_agent_A'][reward_sample_index],
            run_properties['discount_rate_agent_A']
        )

        return {
            'mdp_graph': transition_graphs[0],
            'seed_policy_graph_A': transition_graphs[1],
            'policy_graph_H': policy_graph_H,
            'reward_function_H': reward_function_H,
            'policy_graph_A': policy_tensor_to_graph(
                learn.find_optimal_policy(
                    reward_function_A,
                    discount_rate_A,
                    graph.graphs_to_full_transition_tensor(
                        *transition_graphs, acting_agent_is_H=False, return_sparse=False
                    )
                ),
                transition_graphs[0],
                acting_agent_is_H=False
            ),
            'reward_function_A': reward_function_A
        }

def next_state_rollout(current_state, mdp_graph, policy_graph_H, policy_graph_A=None):
    state_list = graph.get_states_from_graph(mdp_graph)

    state_vector = graph.state_to_vector(current_state, state_list)
    policy_tensor_H = graph.graph_to_policy_tensor(policy_graph_H, return_sparse=False)

    full_transition_tensor = graph.graph_to_full_transition_tensor(mdp_graph, return_sparse=False) if (
        policy_graph_A is None
    ) else graph.graphs_to_full_transition_tensor(
        mdp_graph, policy_graph_A, acting_agent_is_H=True, return_sparse=False
    )

    return dist.sample_from_state_list(
        state_list,
        graph.one_step_rollout(
            graph.compute_state_transition_matrix(full_transition_tensor, policy_tensor_H),
            state_vector
        )
    )

def simulate_policy_rollout(
    initial_state,
    policy_graph_H,
    mdp_graph,
    policy_graph_A=None,
    number_of_steps=20,
    random_seed=0
):
    check.check_state_in_graph_states(mdp_graph, initial_state)
    check.check_policy_graph(policy_graph_H)

    if policy_graph_A is None:
        check.check_mdp_graph(mdp_graph)
        check.check_full_graph_compatibility(mdp_graph, policy_graph_H)
    
    else:
        check.check_joint_mdp_graph(mdp_graph)
        check.check_policy_graph(policy_graph_A)
        check.check_joint_mdp_and_policy_compatibility(mdp_graph, policy_graph_H, policy_is_for_agent_H=True)
        check.check_joint_mdp_and_policy_compatibility(mdp_graph, policy_graph_A, policy_is_for_agent_H=False)
    
    misc.set_global_random_seed(random_seed)

    state_rollout_ = [initial_state]

    for _ in range(number_of_steps):
        state_rollout_ += [
            next_state_rollout(state_rollout_[-1], mdp_graph, policy_graph_H, policy_graph_A=policy_graph_A)
        ]

    return state_rollout_
