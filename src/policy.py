import copy as cp
import networkx as nx

from .utils import graph
from . import check

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