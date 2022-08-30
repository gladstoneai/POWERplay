import copy as cp
import networkx as nx

from .lib.utils import graph
from .lib.utils import dist
from .lib.utils import misc
from .lib import check
from .lib import runex
from . import multi
from . import mdp

def single_agent_to_multiagent_policy_graph(single_agent_policy_graph, current_agent_is_A=True):
    multiagent_graph_ = nx.DiGraph()

    for other_agent_state in graph.get_states_from_graph(single_agent_policy_graph):
        multiagent_graph_ = nx.compose(
            multiagent_graph_,
            nx.relabel_nodes(cp.deepcopy(single_agent_policy_graph), {
                node: multi.single_agent_to_multiagent_graph_node(
                    node, other_agent_state, current_agent_is_A=current_agent_is_A
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

def create_simultaneous_multiagent_random_policy(mdp_graph, current_agent_is_A=True):
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
            ] if current_agent_is_A else [
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

def quick_mdp_to_policy(mdp_graph, current_agent_is_A=True):
    if graph.are_general_graph_states_multiagent(graph.get_states_from_graph(mdp_graph)):
        return create_simultaneous_multiagent_random_policy(mdp_graph, current_agent_is_A=current_agent_is_A)

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

# Note: associated_mdp_graph must be in stochastic format, and have state
def policy_tensor_to_graph(policy_tensor, associated_mdp_graph):
    policy_graph_ = graph.quick_mdp_to_policy(associated_mdp_graph)
    state_list = graph.get_states_from_graph(associated_mdp_graph)
    action_list = graph.get_actions_from_graph(associated_mdp_graph)

    for state in state_list:
        policy_graph_ = update_state_actions(policy_graph_, state, {
            action: float(
                policy_tensor[state_list.index(state)][action_list.index(action)]
            ) for action in graph.get_available_actions_from_graph_state(associated_mdp_graph, state)
        })

    return policy_graph_

# TODO: Document.
def sample_optimal_policy_from_run(run_properties, reward_sample_index=0):

    reward_function_A, discount_rate_A, transition_graphs, convergence_threshold, sweep_type = (
        run_properties['reward_samples'][reward_sample_index],
        run_properties['discount_rate'],
        run_properties['transition_graphs'],
        run_properties['convergence_threshold'],
        run_properties['sweep_type']
    )

    policy_graph_A = policy_tensor_to_graph(
        runex.find_optimal_policy(
            reward_function_A,
            discount_rate_A,
            graph.any_graphs_to_transition_tensor(*transition_graphs),
            value_initialization=None,
            convergence_threshold=convergence_threshold
        ), # NOTE: This works for the agent of a single-agent system, OR for Agent A of a multi-agent system.
        transition_graphs[0]
    )

    if sweep_type == 'single_agent':
        return {
            'mdp_graph_A': transition_graphs[0],
            'policy_graph_A': policy_graph_A,
            'reward_function_A': reward_function_A
        }
    
    elif sweep_type == 'multiagent_fixed_policy':
        return {
            'mdp_graph_A': transition_graphs[0],
            'policy_graph_A': policy_graph_A,
            'reward_function_A': reward_function_A,
            'mdp_graph_B': transition_graphs[2],
            'policy_graph_B': transition_graphs[1]
        }

    elif sweep_type == 'multiagent_with_reward': # transition_graphs = (mdp_graph_A, mdp_graph_B)
        reward_function_B, discount_rate_B = (
            run_properties['reward_samples_agent_B'][reward_sample_index],
            run_properties['discount_rate_agent_B']
        )

        return {
            'mdp_graph_A': transition_graphs[0],
            'seed_policy_graph_B': transition_graphs[1],
            'policy_graph_A': policy_graph_A,
            'reward_function_A': reward_function_A,
            'mdp_graph_B': transition_graphs[2],
            'policy_graph_B': policy_tensor_to_graph(
                runex.find_optimal_policy(
                    reward_function_B,
                    discount_rate_B,
                    graph.graphs_to_multiagent_transition_tensor(
                        transition_graphs[2], policy_graph_A, transition_graphs[0] # Note that the ordering is mdp_graph_B, policy_graph_A, mdp_graph_A because we want to calculate the Agent B policy
                    )
                ),
                transition_graphs[2]
            ),
            'reward_function_B': reward_function_B
        }

def next_state_rollout(current_state, policy_graph, mdp_graph):
    state_list = graph.get_states_from_graph(policy_graph)
    return dist.sample_from_state_list(
        state_list,
        graph.one_step_rollout(
            graph.state_to_vector(current_state, state_list),
            graph.graph_to_policy_tensor(policy_graph),
            graph.graph_to_transition_tensor(mdp_graph)
        )
    )

def simulate_policy_rollout(
    initial_state,
    policy_graph_A,
    mdp_graph_A,
    policy_graph_B=None,
    mdp_graph_B=None,
    number_of_steps=20,
    random_seed=0
):
    check.check_full_graph_compatibility(policy_graph_A, mdp_graph_A)
    check.check_state_in_graph_states(mdp_graph_A, initial_state)

    if policy_graph_B is not None:
        check.check_full_graph_compatibility(policy_graph_B, mdp_graph_B)
        check.check_graph_state_compatibility(mdp_graph_A, mdp_graph_B)
    
    misc.set_global_random_seed(random_seed)
    
    state_rollout_ = [initial_state]
    
    for _ in range(number_of_steps):
        state_rollout_ += [next_state_rollout(state_rollout_[-1], policy_graph_A, mdp_graph_A)]

        if policy_graph_B is not None:
            state_rollout_ += [next_state_rollout(state_rollout_[-1], policy_graph_B, mdp_graph_B)]
    
    return state_rollout_
