import matplotlib.pyplot as plt
import torch
import copy as cp

from .lib.utils import graph
from .lib import data
from .lib import get
from . import mdp
from . import policy
from . import multi
from . import viz

def update_mdp_graph_with_interface(input_mdp, agent_label='A'):
    output_mdp_ = cp.deepcopy(input_mdp)

    print('Now updating MDP for Agent {}.'.format(agent_label))
    print(
        'For each state, input the new allowed actions you want from that state, one at a time. ' \
        'Press Enter to skip and keep the allowed actions as they are.'
    )
    print(
        'For each action of each state, input the state that action will take you to. ' \
        'The state you type will get assigned probability 1, the rest 0.'
    )
    print()

    for multi_state in graph.get_states_from_graph(input_mdp):
        first_check = True
        new_actions_ = []
        print()

        while first_check or new_action_:
            first_check = False
            new_action_ = input(
                'State \'{0}\' currently allows actions {1} for agent {2}. ' \
                'Input new actions, one at a time (Enter to skip): '.format(
                    multi_state,
                    ', '.join([
                        '\'{}\''.format(action) for action in graph.get_available_actions_from_graph_state(
                            input_mdp, multi_state
                        )
                    ]),
                    agent_label
                )
            )

            if new_action_:
                new_actions_ += [new_action_]
        
        new_next_states_dicts_ = []
        print()

        for new_action in new_actions_:
            if new_action in graph.get_available_actions_from_graph_state(input_mdp, multi_state):
                print(
                    'Action \'{0}\' from state \'{1}\' currently has the following next-state probabilities:'.format(new_action, multi_state)
                )
                print(graph.get_available_states_and_probabilities_from_mdp_graph_state_and_action(input_mdp, multi_state, new_action))
            
            new_next_state_ = input(
                'Input new next state for action \'{0}\' from state \'{1}\'{2}: '.format(
                    new_action,
                    multi_state,
                    ' (Enter to skip)' if new_action in graph.get_available_actions_from_graph_state(input_mdp, multi_state) else ''
                )
            )

            new_next_states_dicts_ += [
                { new_next_state_: 1 } if new_next_state_ else graph.get_available_states_and_probabilities_from_mdp_graph_state_and_action(
                    input_mdp, multi_state, new_action
                )
            ]

        if len(new_actions_) > 0:
            output_mdp_ = mdp.update_state_action(
                output_mdp_,
                multi_state,
                { action: next_state_dict for action, next_state_dict in zip(new_actions_, new_next_states_dicts_) },
                check_closure=True
            )
    
    return output_mdp_

def plot_alignment_curves(sweep_id, agent_B_baseline_power=None, plot_title='', folder=data.EXPERIMENT_FOLDER):
    run_suffixes = get.get_sweep_run_suffixes_for_param(sweep_id, 'reward_correlation', folder=folder)

    all_run_props_ = []

    print()

    for run_suffix in run_suffixes:

        print('Accessing run {}...'.format(run_suffix))
        all_run_props_ += [get.get_properties_from_run(sweep_id, run_suffix=run_suffix)]
    
    print()

    all_correlations = [run_props['reward_correlation'] for run_props in all_run_props_]
    all_powers_A = [run_props['power_samples'].mean() for run_props in all_run_props_]
    all_powers_B = [run_props['power_samples_agent_B'].mean() for run_props in all_run_props_]

    _, ax = plt.subplots()

    ax.plot(all_correlations, all_powers_A, 'b.', label='Agent A POWER')
    ax.plot(all_correlations, all_powers_B, 'r.', label='Agent B POWER')
    ax.set_xlabel('Reward correlation')
    ax.set_ylabel('POWER')
    ax.set_ylim([0, 1])
    ax.set_title(plot_title)

    if agent_B_baseline_power is not None:
        ax.plot(all_correlations, [agent_B_baseline_power] * len(all_correlations), 'r-', label='Agent B baseline POWER')

    ax.legend()

    plt.show()

def plot_specific_alignment_curves(
    sweep_id,
    show=True,
    fig_name='',
    expt_folder=data.EXPERIMENT_FOLDER,
    save_folder=data.TEMP_FOLDER
):
    graph_buffer = 0.1

    print('Accessing sweep {}...'.format(sweep_id))

    run_suffixes = get.get_sweep_run_suffixes_for_param(sweep_id, 'reward_correlation', folder=expt_folder)
    all_run_props = [get.get_properties_from_run(sweep_id, run_suffix=run_suffix) for run_suffix in run_suffixes]
    all_correlations = [run_props['reward_correlation'] for run_props in all_run_props]
    all_powers_A = [run_props['power_samples'].mean(dim=0) for run_props in all_run_props]
    all_powers_B = [run_props['power_samples_agent_B'].mean(dim=0) for run_props in all_run_props]

    min_A_power = min([powers_A.min() for powers_A in all_powers_A]) * (1 - graph_buffer)
    max_A_power = max([powers_A.max() for powers_A in all_powers_A]) * (1 + graph_buffer)
    min_B_power = min([powers_B.min() for powers_B in all_powers_B]) * (1 - graph_buffer)
    max_B_power = max([powers_B.max() for powers_B in all_powers_B]) * (1 + graph_buffer)

    correlation_coefficients = []

    for correlation, power_samples_A, power_samples_B in zip(all_correlations, all_powers_A, all_powers_B):
        fig, ax = plt.subplots()

        ax.plot(power_samples_A, power_samples_B, 'b.')
        ax.set_xlabel('State POWER values, Agent A')
        ax.set_ylabel('State POWER values, Agent B')
        ax.set_title('Reward correlation value {}'.format(correlation))
        ax.set_xlim([min_A_power, max_A_power])
        ax.set_ylim([min_B_power, max_B_power])

        if fig_name:
            data.save_figure(
                fig,
                '{0}-sweep_id_{1}-specific_alignment_curve-correlation_{2}'.format(fig_name, sweep_id, str(correlation)),
                folder=save_folder
            )

        correlation_coefficients += [torch.corrcoef(torch.stack([power_samples_A, power_samples_B]))[0][1]]

    fig, ax = plt.subplots()

    ax.plot(all_correlations, correlation_coefficients, 'r.')
    ax.set_xlabel('Reward correlation value')
    ax.set_ylabel('State-by-state POWER correlation value')
    ax.set_title('Correlation coefficients plot')

    if fig_name:
        data.save_figure(
            fig,
            '{0}-sweep_id_{1}-specific_alignment_curve-correlation_FULL'.format(fig_name, sweep_id),
            folder=save_folder
        )

    if show:
        plt.show()

def construct_single_agent_gridworld_mdp(num_rows, num_cols, mdp_save_name=None):
    stochastic_graph = mdp.gridworld_to_stochastic_graph(
            mdp.construct_gridworld(
                num_rows, num_cols, name='{0}x{1} gridworld'.format(num_rows, num_cols)
            )
        )
    
    if mdp_save_name is not None:
        data.save_graph_to_dot_file(stochastic_graph, mdp_save_name, folder=data.MDPS_FOLDER)
    
    return stochastic_graph

def construct_multiagent_gridworld_policy_and_mdps(num_rows, num_cols, mdp_save_name=None, policy_save_name=None):
    stochastic_graph = mdp.gridworld_to_stochastic_graph(
            mdp.construct_gridworld(
                num_rows, num_cols, name='{0}x{1} gridworld'.format(num_rows, num_cols)
            )
        )
    policy_b_ = graph.quick_mdp_to_policy(stochastic_graph)

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

def construct_multiagent_gridworld_mdps_with_interactions(num_rows, num_cols, mdp_save_name=None):
    stochastic_graph = mdp.gridworld_to_stochastic_graph(
        mdp.construct_gridworld(
            num_rows, num_cols, name='{0}x{1} gridworld'.format(num_rows, num_cols)
        )
    )

    mdp_A = update_mdp_graph_with_interface(
        multi.create_multiagent_graph(stochastic_graph, current_agent_is_A=True), agent_label='A'
    )
    mdp_B = update_mdp_graph_with_interface(
        multi.create_multiagent_graph(stochastic_graph, current_agent_is_A=False), agent_label='B'
    )

    if mdp_save_name is not None:
        data.save_graph_to_dot_file(mdp_A, '{}_agent_A'.format(mdp_save_name), folder=data.MDPS_FOLDER)
        data.save_graph_to_dot_file(mdp_B, '{}_agent_B'.format(mdp_save_name), folder=data.MDPS_FOLDER)
    
    return [
        mdp_A,
        mdp_B
    ]

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

def build_quick_random_policy(mdp_graph):
    return graph.quick_mdp_to_policy(mdp_graph)