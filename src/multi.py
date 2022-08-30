import networkx as nx
import itertools as it

from .lib.utils import graph
from . import mdp

def single_agent_to_multiagent_graph_node(
    current_agent_graph_node,
    other_agent_state,
    current_agent_is_A=True
):
    states_and_actions_single = graph.decompose_stochastic_graph_node(current_agent_graph_node)
    action_index = 0 if len(states_and_actions_single) == 2 else 1

    return graph.build_stochastic_graph_node(*[
        graph.single_agent_states_to_multiagent_state(*([
            states_and_actions_single[i], other_agent_state
        ] if current_agent_is_A else [
            other_agent_state, states_and_actions_single[i]
        ])) if (
            i != action_index
        ) else states_and_actions_single[i] for i in range(len(states_and_actions_single))
    ])

def create_simultaneous_multiagent_graph(single_agent_graph):
    multiagent_graph_ = nx.DiGraph()

    state_list = graph.get_states_from_graph(single_agent_graph)

    for joint_state in [
        graph.single_agent_states_to_multiagent_state(*state_pair) for state_pair in it.product(
            state_list, state_list
        )
    ]:
        state_A, state_B = graph.multiagent_state_to_single_agent_states(joint_state)
        available_actions_A = graph.get_available_actions_from_graph_state(single_agent_graph, state_A)
        available_actions_B = graph.get_available_actions_from_graph_state(single_agent_graph, state_B)

        action_dict_ = {}

        for joint_action in [
            graph.single_agent_actions_to_multiagent_action(*action_pair) for action_pair in it.product(
                available_actions_A, available_actions_B
            )
        ]:

            action_A, action_B = graph.multiagent_action_to_single_agent_actions(joint_action)

            next_states_A = list(
                graph.get_available_states_and_probabilities_from_mdp_graph_state_and_action(
                    single_agent_graph, state_A, action_A
                ).items()
            )
            next_states_B = list(
                graph.get_available_states_and_probabilities_from_mdp_graph_state_and_action(
                    single_agent_graph, state_B, action_B
                ).items()
            )

            action_dict_[joint_action] = {
                graph.single_agent_states_to_multiagent_state(
                    next_joint_state[0][0], next_joint_state[1][0]
                ): next_joint_state[0][1] * next_joint_state[1][1] for next_joint_state in it.product(
                    next_states_A, next_states_B
                )
            }
        
        multiagent_graph_ = mdp.add_state_action(
            multiagent_graph_, joint_state, action_dict_, check_closure=False
        )

    return multiagent_graph_
