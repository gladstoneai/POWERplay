from . import mdp
from . import policy
from . import multi
from . import viz
from .lib.utils import graph
from .lib import data

def construct_gridworld_policy_and_mdps(num_rows, num_cols, mdp_save_name=None, policy_save_name=None):
    stochastic_graph = mdp.gridworld_to_stochastic_graph(
            mdp.construct_gridworld(
                num_rows, num_cols, name='{0}x{1} gridworld'.format(num_rows, num_cols)
            )
        )
    policy_b_ = policy.quick_mdp_to_policy(stochastic_graph)

    print('Now updating policy for Agent B.')
    print('State update example: { \'stay\': 0, \'up\': 1, \'down\': 0, \'left\': 0, \'right\': 0 }')
    print('The action you type (stay/up/down/left/right) gets set to 1, all others to 0.')
    print()

    for state in graph.get_states_from_graph(policy_b_):
        action_to_take = input('Enter action for individual state \'{}\': '.format(state)).replace('\'', '"')

        if action_to_take:
            policy_b_ = policy.update_state_actions(
                policy_b_,
                state,
                {
                    action: (
                        1 if action == action_to_take else 0
                    ) for action in graph.get_available_actions_from_graph_state(policy_b_, state)
                }
            )
    
    policy_b_multi_ = multi.create_multiagent_graph(policy_b_, current_agent_is_A=False)

    print()

    for multi_state in graph.get_states_from_graph(policy_b_multi_):
        action_to_take = input('Enter action for multiagent state \'{}\': '.format(multi_state)).replace('\'', '"')

        if action_to_take:
            policy_b_multi_ = policy.update_state_actions(
                policy_b_multi_,
                multi_state,
                {
                    action: (
                        1 if action == action_to_take else 0
                    ) for action in graph.get_available_actions_from_graph_state(policy_b_multi_, multi_state)
                }
            )

    print()

    mdp_A = multi.create_multiagent_graph(stochastic_graph, current_agent_is_A=True)
    mdp_B = multi.create_multiagent_graph(stochastic_graph, current_agent_is_A=False)

    if mdp_save_name is not None:
        data.save_graph_to_dot_file(mdp_A, '{}_agent_A'.format(mdp_save_name), folder=data.MDPS_FOLDER)
        data.save_graph_to_dot_file(mdp_B, '{}_agent_B'.format(mdp_save_name), folder=data.MDPS_FOLDER)
    
    if policy_save_name is not None:
        data.save_graph_to_dot_file(
            policy_b_multi_,
            '{0}_agent_B_{1}'.format(mdp_save_name, policy_save_name),
            folder=data.POLICIES_FOLDER
        )

    return {
        'policy_B': policy_b_multi_,
        'mdp_A': mdp_A,
        'mdp_B': mdp_B
    }

def visualize_full_gridworld_rollout(
    sweep_id,
    run_suffix,
    initial_state='(0, 0)_A^(0, 0)_B',
    reward_sample_index=0,
    number_of_steps=20,
    show=True,
    save_handle='gridworld_rollout',
    save_folder=data.TEMP_FOLDER
):
    policy_data = policy.sample_optimal_policy_from_run(
        sweep_id,
        run_suffix,
        reward_sample_index=reward_sample_index
    )

    viz.plot_gridworld_rollout(
        graph.get_states_from_graph(policy_data['outputs']['policy_graph_A']),
        policy.simulate_policy_rollout(
            initial_state,
            policy_data['outputs']['policy_graph_A'],
            policy_data['inputs']['mdp_graph_A'],
            policy_graph_B=policy_data['inputs'].get('policy_graph_B'),
            mdp_graph_B = policy_data['inputs'].get('mdp_graph_B'),
            number_of_steps=number_of_steps
        ),
        reward_function=policy_data['inputs']['reward_function'],
        show=show,
        save_handle=save_handle,
        save_folder=save_folder
    )