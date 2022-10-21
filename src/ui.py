import copy as cp

from .lib.utils import graph
from . import multi
from . import mdp
from . import policy

def update_mdp_graph_interface(input_mdp):
    output_mdp_ = cp.deepcopy(input_mdp)

    print()

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
        'Actions should be in multi-agent format if this is a multi-agent graph, e.g., \'left_H^stay_A\'. '\
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

def update_multiagent_policy_interface(policy_graph):
    policy_A_ = cp.deepcopy(policy_graph)

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
        action_to_take = input('Enter action for multi-agent state \'{}\': '.format(multi_state)).replace('\'', '"')

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

    return policy_A_multi_