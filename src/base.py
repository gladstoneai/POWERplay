import copy as cp

from .lib.utils import dist
from .lib.utils import graph
from .lib import data
from .lib import get
from .lib import check
from . import mdp
from . import policy
from . import multi
from . import view
from . import anim

def update_mdp_graph_with_interface(input_mdp):
    output_mdp_ = cp.deepcopy(input_mdp)

    if graph.is_graph_multiagent(input_mdp):
        delete_overlapping = input(
            'Do you want to delete overlapping states in this MDP? (y/n) '\
            'In a gridworld, this is equivalent to forbidding the agents from occupying the same cell. '
        )
        print()

        if delete_overlapping == 'y':
            output_mdp_ = multi.remove_states_with_overlapping_agents(output_mdp_)

    print(
        'For each state, input the new allowed actions you want from that state, one at a time. '\
        'Actions should be in multiagent format if this is a multiagent graph, e.g., \'left_H^stay_A\'. '\
        'Press Enter to skip and keep the allowed actions as they are.'
    )
    print(
        'For each action of each state, input the state that action will take you to. '\
        'The state you type will get assigned probability 1, the rest 0.'
    )
    print()

    for state in graph.get_states_from_graph(output_mdp_):
        first_check = True
        new_actions_ = []
        print()

        while first_check or new_action_:
            first_check = False
            new_action_ = input(
                'State \'{0}\' currently allows actions {1}. ' \
                'Input new actions, one at a time (Enter to skip): '.format(
                    state,
                    ', '.join([
                        '\'{}\''.format(action) for action in graph.get_available_actions_from_graph_state(
                            output_mdp_, state
                        )
                    ])
                )
            )

            if new_action_:
                new_actions_ += [new_action_]
        
        new_next_states_dicts_ = []
        print()

        for new_action in new_actions_:
            if new_action in graph.get_available_actions_from_graph_state(output_mdp_, state):
                print(
                    'Action \'{0}\' from state \'{1}\' currently has the following next-state probabilities:'.format(
                        new_action, state
                    )
                )
                print(
                    graph.get_available_states_and_probabilities_from_mdp_graph_state_and_action(
                        output_mdp_, state, new_action
                    )
                )
            
            new_next_state_ = input(
                'Input new next state for action \'{0}\' from state \'{1}\'{2}: '.format(
                    new_action,
                    state,
                    ' (Enter to skip)' if new_action in graph.get_available_actions_from_graph_state(output_mdp_, state) else ''
                )
            )

            new_next_states_dicts_ += [
                { new_next_state_: 1 } if (
                    new_next_state_
                ) else graph.get_available_states_and_probabilities_from_mdp_graph_state_and_action(
                    output_mdp_, state, new_action
                )
            ]

        if len(new_actions_) > 0:
            output_mdp_ = mdp.update_state_action(
                output_mdp_,
                state,
                { action: next_state_dict for action, next_state_dict in zip(new_actions_, new_next_states_dicts_) },
                check_closure=True
            )
    
    return output_mdp_

def construct_single_agent_gridworld_mdp(num_rows, num_cols, mdp_save_name=None, squares_to_delete=[]):
    stochastic_graph = mdp.gridworld_to_stochastic_graph(
            mdp.construct_gridworld(
                num_rows, num_cols, name='{0}x{1} gridworld'.format(num_rows, num_cols), squares_to_delete=squares_to_delete
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
    policy_A_ = policy.quick_mdp_to_policy(stochastic_graph)

    print('Now updating policy for Agent A.')
    print('State update example: { \'stay\': 0, \'up\': 1, \'down\': 0, \'left\': 0, \'right\': 0 }')
    print('The action you type (stay/up/down/left/right) gets set to 1, all others to 0.')
    print()

    for state in graph.get_states_from_graph(policy_A_):
        action_to_take = input('Enter action for individual state \'{}\': '.format(state)).replace('\'', '"')

        if action_to_take:
            policy_A_ = policy.update_state_actions(
                policy_A_,
                state,
                {
                    action: (
                        1 if action == action_to_take else 0
                    ) for action in graph.get_available_actions_from_graph_state(policy_A_, state)
                }
            )
    
    policy_A_multi_ = policy.single_agent_to_multiagent_policy_graph(policy_A_, acting_agent_is_H=False)

    print()

    for multi_state in graph.get_states_from_graph(policy_A_multi_):
        action_to_take = input('Enter action for multiagent state \'{}\': '.format(multi_state)).replace('\'', '"')

        if action_to_take:
            policy_A_multi_ = policy.update_state_actions(
                policy_A_multi_,
                multi_state,
                {
                    action: (
                        1 if action == action_to_take else 0
                    ) for action in graph.get_available_actions_from_graph_state(policy_A_multi_, multi_state)
                }
            )

    print()

    multiagent_graph = multi.create_joint_multiagent_graph(stochastic_graph)

    if mdp_save_name is not None:
        data.save_graph_to_dot_file(
            multiagent_graph, mdp_save_name, folder=data.MDPS_FOLDER
        )
    
    if policy_save_name is not None:
        data.save_graph_to_dot_file(
            policy_A_multi_,
            '{0}_agent_A_{1}'.format(mdp_save_name, policy_save_name),
            folder=data.POLICIES_FOLDER
        )

    return {
        'policy_A': policy_A_multi_,
        'mdp': multiagent_graph
    }

def construct_multiagent_gridworld_mdps_with_interactions(num_rows, num_cols, mdp_save_name=None):
    stochastic_graph = mdp.gridworld_to_stochastic_graph(
        mdp.construct_gridworld(
            num_rows, num_cols, name='{0}x{1} gridworld'.format(num_rows, num_cols)
        )
    )

    joint_mdp_graph = update_mdp_graph_with_interface(multi.create_joint_multiagent_graph(stochastic_graph))

    if mdp_save_name is not None:
        data.save_graph_to_dot_file(joint_mdp_graph, mdp_save_name, folder=data.MDPS_FOLDER)
    
    return joint_mdp_graph

# y_axis_bounds is either None (and set the y-axis bounds by default based on the data) or a 2-element list
# with the lower bound as the first element, and the upper bound as the second element.
def visualize_alignment_curves(
    sweep_id,
    show=True,
    plot_title='Alignment curves',
    fig_name='',
    include_baseline_power=True,
    y_axis_bounds=None,
    data_folder=data.EXPERIMENT_FOLDER,
    save_folder=data.TEMP_FOLDER
):
    view.plot_alignment_curves(
        sweep_id,
        show=show,
        plot_title=plot_title,
        include_baseline_power=include_baseline_power,
        fig_name=fig_name,
        y_axis_bounds=y_axis_bounds,
        data_folder=data_folder,
        save_folder=save_folder
    )

def visualize_all_alignment_curves(
    sweep_ids_list,
    show=False,
    plot_titles=None,
    fig_names=None,
    data_folder=data.EXPERIMENT_FOLDER,
    save_folder=data.TEMP_FOLDER
):
    plot_titles_list =  ['Alignment curves'] * len(sweep_ids_list) if plot_titles is None else plot_titles
    fig_names_list = [None] * len(sweep_ids_list) if fig_names is None else fig_names

    for sweep_id, plot_title, fig_name in zip(sweep_ids_list, plot_titles_list, fig_names_list):
        view.plot_alignment_curves(
            sweep_id,
            show=show,
            plot_title=plot_title,
            fig_name=fig_name,
            data_folder=data_folder,
            save_folder=save_folder
        )

def visualize_specific_power_alignments(
    sweep_id,
    show=True,
    ms_per_frame=100,
    fig_name=None,
    include_baseline_powers=True,
    data_folder=data.EXPERIMENT_FOLDER,
    save_folder=data.TEMP_FOLDER
):
    view.plot_specific_power_alignments(
        sweep_id,
        show=show,
        ms_per_frame=ms_per_frame,
        fig_name=fig_name,
        include_baseline_powers=include_baseline_powers,
        data_folder=data_folder,
        save_folder=save_folder
    )

def visualize_full_gridworld_rollout(
    sweep_id,
    run_suffix='',
    initial_state='(0, 0)_H^(0, 0)_A',
    reward_sample_index=0,
    number_of_steps=20,
    agent_whose_rewards_are_displayed='A',
    ms_per_frame=400,
    show=True,
    save_handle='gridworld_rollout',
    save_folder=data.TEMP_FOLDER
):
    check.check_agent_label(agent_whose_rewards_are_displayed)

    run_properties = get.get_properties_from_run(sweep_id, run_suffix=run_suffix)
    policy_data = policy.sample_optimal_policy_data_from_run(run_properties, reward_sample_index=reward_sample_index)

    view.plot_gridworld_rollout(
        graph.get_states_from_graph(policy_data['policy_graph_H']),
        policy.simulate_policy_rollout(
            initial_state,
            policy_data['policy_graph_H'],
            policy_data['mdp_graph'],
            policy_graph_A=policy_data['policy_graph_A'],
            number_of_steps=number_of_steps,
            random_seed=0
        ),
        reward_function=policy_data['reward_function_{}'.format(agent_whose_rewards_are_displayed)],
        ms_per_frame=ms_per_frame,
        show=show,
        agent_whose_rewards_are_displayed=agent_whose_rewards_are_displayed,
        save_handle=save_handle,
        save_folder=save_folder
    )

def visualize_correlated_reward_samples(
    num_samples,
    distribution_config={ 'dist_name': 'uniform', 'params': [0, 1] },
    correlation=0,
    symmetric_interval=None,
    random_seed=0,
    show=True,
    fig_name=None,
    save_folder=data.TEMP_FOLDER
):
    view.plot_correlated_reward_samples(
        num_samples,
        distribution_config=distribution_config,
        correlation=correlation,
        symmetric_interval=symmetric_interval,
        distribution_dict=dist.DISTRIBUTION_DICT,
        random_seed=random_seed,
        show=show,
        fig_name=fig_name,
        save_folder=save_folder
    )

def visualize_all_correlated_reward_samples(
    num_samples,
    distribution_config={ 'dist_name': 'uniform', 'params': [0, 1] },
    correlations_list=[],
    symmetric_interval=None,
    random_seed=0,
    ms_per_frame=100,
    fig_name='correlated_reward_animation',
    save_folder=data.TEMP_FOLDER
):
    fig_filenames_ = []

    for correlation in correlations_list + [correlations_list[-1]] * 4: # This freezes the last frame for a few seconds
        fig_filename = '{0}-correlation_{1}'.format(fig_name, correlation)

        view.plot_correlated_reward_samples(
            num_samples,
            distribution_config=distribution_config,
            correlation=correlation,
            symmetric_interval=symmetric_interval,
            distribution_dict=dist.DISTRIBUTION_DICT,
            random_seed=random_seed,
            show=False,
            fig_name=fig_filename,
            save_folder=data.TEMP_FOLDER
        )

        fig_filenames_ += [fig_filename]

    anim.animate_from_filenames(
        fig_filenames_,
        fig_name,
        ms_per_frame=ms_per_frame,
        input_folder_or_list=data.TEMP_FOLDER,
        output_folder=save_folder
    )

def visualize_power_relationship_over_multiple_sweeps(
    x_axis_values,
    sweep_ids_list,
    show=True,
    fig_name='temp',
    x_axis_label='name of variable',
    save_folder=data.TEMP_FOLDER
):
    power_correlation_data = [
        get.get_reward_correlations_and_powers_from_sweep(sweep_id, include_baseline_power=False) for (
            sweep_id
        ) in sweep_ids_list
    ]

    all_powers_H_ = []
    all_powers_A_ = []

    for correlation_item in power_correlation_data:
        all_powers_H_ += correlation_item['all_powers_H']
        all_powers_A_ += correlation_item['all_powers_A']
    
    view.plot_power_correlation_relationship(
        x_axis_values,
        all_powers_H_,
        all_powers_A_,
        x_axis_label=x_axis_label,
        show=show,
        fig_name=fig_name,
        save_folder=save_folder
    )

def build_quick_random_policy(mdp_graph):
    return policy.quick_mdp_to_policy(mdp_graph)
