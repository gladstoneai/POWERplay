import copy as cp
import networkx as nx

from .lib.utils import graph
from .lib.utils import dist
from .lib.utils import misc
from .lib import data
from .lib import check
from .lib import get
from .lib import runex

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
def sample_optimal_policy_from_run(
    sweep_id,
    run_suffix,
    reward_sample_index=0,
    mdps_folder=data.MDPS_FOLDER,
    policies_folder=data.POLICIES_FOLDER
):
    run_properties = get.get_properties_from_run(
        sweep_id, run_suffix=run_suffix, mdps_folder=mdps_folder, policies_folder=policies_folder
    )
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
            graph.any_graphs_to_transition_tensor(*[transition_graphs[0]] + ([] if (
                sweep_type != 'multiagent_with_reward'
            ) else [graph.quick_mdp_to_policy(transition_graphs[1]), transition_graphs[1]])), # This creates a uniform random policy for Agent A
            value_initialization=None,
            convergence_threshold=convergence_threshold
        ), # NOTE: This works for the agent of a single-agent system, OR for Agent A of a multi-agent system.
        transition_graphs[0]
    )

    if sweep_type == 'single_agent':
        multiagent_dict = {}
    
    elif sweep_type == 'multiagent_fixed_policy':
        multiagent_dict = {
            'policy_graph_B': transition_graphs[1],
            'mdp_graph_B': transition_graphs[2]
        }
    
    elif sweep_type == 'multiagent_with_reward': # transition_graphs = (mdp_graph_A, mdp_graph_B)
        reward_function_B, discount_rate_B = (
            run_properties['reward_samples_agent_B'][reward_sample_index],
            run_properties['discount_rate_agent_B']
        )

        multiagent_dict = {
            'policy_graph_B': policy_tensor_to_graph(
                runex.find_optimal_policy(
                    reward_function_B,
                    discount_rate_B,
                    graph.graphs_to_multiagent_transition_tensor(
                        transition_graphs[1], policy_graph_A, transition_graphs[0] # Note that the ordering is mdp_graph_B, policy_graph_A, mdp_graph_A because we want to calculate the Agent B policy
                    )
                ),
                transition_graphs[1]
            ),
            'mdp_graph_B': transition_graphs[1],
            'reward_function_B': reward_function_B,
            'discount_rate_B': discount_rate_B
        }

    return {
        'inputs': {
            'mdp_graph_A': transition_graphs[0],
            'reward_function': reward_function_A,
            'discount_rate': discount_rate_A,
            'convergence_threshold': convergence_threshold,
            **multiagent_dict
        },
        'outputs': {
            'policy_graph_A': policy_graph_A
        }
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
