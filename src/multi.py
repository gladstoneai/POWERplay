import networkx as nx
import copy as cp

from .utils import graph

def single_agent_to_multiagent_state(
    current_agent_state,
    other_agent_state,
    current_agent_is_A=True
): 
    return '{0}_A^{1}_B'.format(
        *([
            current_agent_state, other_agent_state
        ] if current_agent_is_A else [
            other_agent_state, current_agent_state
        ])
    )

def single_agent_to_multiagent_graph_node(
    current_agent_graph_node,
    other_agent_state,
    current_agent_is_A=True
):
    states_and_actions_single = graph.decompose_stochastic_graph_node(current_agent_graph_node)
    action_index = 0 if len(states_and_actions_single) == 2 else 1

    return graph.build_stochastic_graph_node(*[
        single_agent_to_multiagent_state(
            states_and_actions_single[i], other_agent_state, current_agent_is_A=current_agent_is_A
        ) if (
            i != action_index
        ) else states_and_actions_single[i] for i in range(len(states_and_actions_single))
    ])

def create_multiagent_transition_graph(single_agent_graph, current_agent_is_A=True):
    multiagent_graph_ = nx.DiGraph()

    for other_agent_state in graph.get_states_from_graph(single_agent_graph):
        multiagent_graph_ = nx.compose(
            multiagent_graph_, nx.relabel_nodes(cp.deepcopy(single_agent_graph), {
                node: single_agent_to_multiagent_graph_node(
                    node, other_agent_state, current_agent_is_A=current_agent_is_A
                ) for node in single_agent_graph.nodes
            })
        )
    
    return multiagent_graph_