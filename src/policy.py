import copy as cp
import networkx as nx

from .lib.utils import graph
from .lib import data
from .lib import check
from .lib import get
from .lib import learn

def quick_mdp_to_policy(mdp_graph):
    policy_graph_ = cp.deepcopy(mdp_graph)

    policy_graph_.remove_nodes_from([
        node for node in policy_graph_.nodes if len(graph.decompose_stochastic_graph_node(node)) == 3
    ])

    for state_node in graph.get_states_from_graph(policy_graph_):
        state_edges = policy_graph_.edges(state_node)

        nx.set_edge_attributes(
            policy_graph_,
            { edge: (1 / len(state_edges)) for edge in state_edges },
            name='weight'
        )

    return policy_graph_

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

# TODO: Document.
def sample_optimal_policy_from_run(
    sweep_id,
    run_suffix,
    reward_sample_index=0,
    mdps_folder=data.MDPS_FOLDER,
    policies_folder=data.POLICIES_FOLDER
):
    run_properties = get.get_properties_from_run(
        sweep_id, run_suffix, mdps_folder=mdps_folder, policies_folder=policies_folder
    )
    reward_function, discount_rate, transition_graphs, convergence_threshold = (
        run_properties['reward_samples'][reward_sample_index],
        run_properties['discount_rate'],
        run_properties['transition_graphs'],
        run_properties['convergence_threshold']
    )

    return [
        policy_tensor_to_graph(
            learn.compute_optimal_policy_tensor(
                reward_function,
                discount_rate,
                graph.any_graphs_to_transition_tensor(*transition_graphs),
                value_initialization=None,
                convergence_threshold=convergence_threshold
            ),
            transition_graphs[0]
        ), # NOTE: This works for the agent of a single-agent system, OR for Agent A of a multi-agent system.
        reward_function,
        discount_rate
    ] 

# TODO: Finish this function
def simulate_policy_rollout(policy_graph_A, policy_graph_B, initial_state, number_of_steps=20):
    check.check_graph_state_compatibility(policy_graph_A, policy_graph_B)
