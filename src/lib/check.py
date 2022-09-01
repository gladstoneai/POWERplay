import multiprocessing as mps
import warnings as warn
import torch

from .utils import dist
from .utils import graph
from . import data

################################################################################

PROBABILITY_TOLERANCE = 1e-4

################################################################################

def check_project_exists(project, entity, wb_api):
    if project not in [proj.name for proj in wb_api.projects(entity=entity)]:
        raise Exception(
            'The project {0} doesn\'t exist in the entity {1}. '\
            'You need to create a project of that name before running this sweep.'.format(project, entity)
        )

def check_agent_label(agent_label):
    if agent_label != 'A' and agent_label != 'B':
        raise Exception('Agent label must be either \'A\' or \'B\', not {}.'.format(agent_label))

def check_num_samples(num_samples, num_workers):
    if num_samples % num_workers != 0:
        raise Exception('The number of reward samples must be an exact multiple of the number of workers.')

def check_dict_contains_key(dict_to_check, key, dict_name):
    if key not in dict_to_check.keys():
        raise Exception('The dict \'{0}\' must contain a key \'{1}\'.'.format(dict_name, key))

def check_state_in_graph_states(input_graph, state):
    if state not in graph.get_states_from_graph(input_graph):
        raise Exception('The state {} isn\'t in your input_graph graph.'.format(state))

def check_policy_actions(policy_graph, state, policy_actions, tolerance=PROBABILITY_TOLERANCE):
    if set(policy_actions.keys()) != set(graph.get_available_actions_from_graph_state(policy_graph, state)):
        raise Exception(
            'The actions in the update set for state {} ' \
            'must be identical to that state\'s action set in the original policy.'.format(state)
        )
    
    if abs(sum(policy_actions.values()) - 1) > tolerance:
        raise Exception('The probabilities of all actions from state {} must sum to 1.'.format(state))

def check_state_list_identity(state_list_1, state_list_2):
    if state_list_1 != state_list_2:
        raise Exception('The two input graphs must have the same state set.')

# graph_1 and graph_2 could be policy or MDP graphs
def check_graph_state_compatibility(graph_1, graph_2):
    check_state_list_identity(graph.get_states_from_graph(graph_1), graph.get_states_from_graph(graph_2))

# graph_1 and graph_2 could be policy or single-agent MDP graphs
def check_full_graph_compatibility(graph_1, graph_2):
    check_graph_state_compatibility(graph_1, graph_2)
    
    if graph.get_actions_from_graph(graph_1) != graph.get_actions_from_graph(graph_2):
        raise Exception('The two input graphs must have the same action set.')
    
    for state in graph.get_states_from_graph(graph_1):
        if set([
            graph.decompose_stochastic_graph_node(node)[0] for node in graph_1.successors(
                graph.build_stochastic_graph_node(state)
            )
        ]) != set([
            graph.decompose_stochastic_graph_node(node)[0] for node in graph_2.successors(
                graph.build_stochastic_graph_node(state)
            )
        ]):
            raise Exception(
                'The action set for graph_1 and graph_2 must be identical at state {}.'.format(state)
            )

def check_joint_mdp_and_policy_compatibility(joint_mdp_graph, policy_graph, policy_is_for_agent_A=True):
    check_graph_state_compatibility(policy_graph, joint_mdp_graph)

    if policy_is_for_agent_A:
        if graph.get_actions_from_graph(policy_graph) != graph.get_single_agent_actions_from_joint_mdp_graph(joint_mdp_graph)[0]:
            raise Exception('The Agent A policy graph must have the same action set as Agent A does in the joint MDP graph.')
    
    else:
        if graph.get_actions_from_graph(policy_graph) != graph.get_single_agent_actions_from_joint_mdp_graph(joint_mdp_graph)[1]:
            raise Exception('The Agent B policy graph must have the same action set as Agent B does in the joint MDP graph.')
    
    for state in graph.get_states_from_graph(joint_mdp_graph):
        agent_A_actions, agent_B_actions = graph.get_unique_single_agent_actions_from_joint_actions(
            graph.get_available_actions_from_graph_state(joint_mdp_graph, state)
        )

        if policy_is_for_agent_A:
            if graph.get_available_actions_from_graph_state(policy_graph, state) != agent_A_actions:
                raise Exception(
                    'The policy graph and MDP graph have different available actions for Agent A at state \'{}\'.'.format(state)
                )
            
        else:
            if graph.get_available_actions_from_graph_state(policy_graph, state) != agent_B_actions:
                raise Exception(
                    'The policy graph and MDP graph have different available actions for Agent B at state \'{}\'.'.format(state)
                )

def check_stochastic_state_name(name):
    if '__' in name:
        raise Exception(
            'The state or action {} has a double underscore in its name, which is forbidden.'.format(name)
        )

    if ']' in name or '[' in name:
        raise Exception('The state or action {} can\'t have a [ or ] character in its name.'.format(name))
    
    if not graph.are_general_graph_states_multiagent([name]):
        if '_' in name:
            raise Exception(
                'The single-agent state or action {} has an underscore in its name, which is forbidden.'.format(
                    name
                )
            )

def check_action_dict(action_dict, tolerance=PROBABILITY_TOLERANCE):
    if action_dict == {}:
        raise Exception('The action_dict can\'t be empty.')

    for action in action_dict.keys():
        check_stochastic_state_name(action)

        if action_dict[action] == {}:
            raise Exception('The state dict for action {} can\'t be empty.'.format(action))

        if abs(sum([action_dict[action][state] for state in action_dict[action].keys()]) - 1) > tolerance:
            raise Exception('The state transition probabilities for action {} must sum to 1.'.format(action))
        
        for state in action_dict[action].keys():
            check_stochastic_state_name(state)

def check_stochastic_mdp_closure(stoch_mdp_graph):
    out_states_list = list(set([
        graph.decompose_stochastic_graph_node(node)[0] for node in list(
            stoch_mdp_graph
        ) if (len(graph.decompose_stochastic_graph_node(node)) == 3)
    ]))

    for out_state in out_states_list:
        if out_state not in graph.get_states_from_graph(stoch_mdp_graph):
            warn.warn(
                'Your MDP is not closed so can\'t yet be used for experiments. '\
                'State {} exists as an action output but hasn\'t had its accessible actions defined.'.format(
                    out_state
                )
            )

def check_stochastic_noise_level(stochastic_noise_level):
    if stochastic_noise_level < 0 or stochastic_noise_level > 1:
        raise Exception('Stochastic noise level must be between 0 and 1.')

def check_noise_bias(noise_bias, stochastic_noise_level):
    if sum([noise for noise in noise_bias.values()]) > stochastic_noise_level:
        raise Exception('The total noise bias can\'t be greater than the stochastic_noise_level of {}.'.format(
            stochastic_noise_level
        ))
    
    if any([noise < 0 for noise in noise_bias.values()]):
        raise Exception('Every value of the noise bias must be greater than 0.')

def check_mdp_graph(mdp_graph, tolerance=PROBABILITY_TOLERANCE):
    transition_tensor = graph.graph_to_full_transition_tensor(mdp_graph)
    state_list, action_list = graph.get_states_from_graph(mdp_graph), graph.get_actions_from_graph(mdp_graph)

    if list(transition_tensor.shape) != [len(state_list), len(action_list), len(state_list)]:
        raise Exception('The full transition tensor for this MDP must have shape [{0}, {1}, {0}].'.format(
            len(state_list), len(action_list)
        ))
    
    for state_tensor in transition_tensor:
        for action_tensor in state_tensor:
            if (not (action_tensor == 0).all()) and (action_tensor.sum() - 1).abs() > tolerance:
                raise Exception(
                    'Every inner row of the full transition tensor must either be all zeros ' \
                    '(if the action can\'t be taken) or sum to 1 ' \
                    '(the total probability of ending up in any downstream state).'
                )

def check_mdp_graph_by_key(mdp_key, tolerance=PROBABILITY_TOLERANCE, mdps_folder=data.MDPS_FOLDER):
    check_mdp_graph(data.load_graph_from_dot_file(mdp_key, folder=mdps_folder), tolerance=tolerance)

def check_joint_mdp_graph(joint_mdp_graph, tolerance=PROBABILITY_TOLERANCE):
    try:
        joint_transition_tensor = graph.graph_to_joint_transition_tensor(joint_mdp_graph)
    except IndexError:
        raise Exception('This MDP graph should be a joint MDP graph.')

    state_list = graph.get_states_from_graph(joint_mdp_graph)
    action_list_A, action_list_B = graph.get_single_agent_actions_from_joint_mdp_graph(joint_mdp_graph)

    if list(joint_transition_tensor.shape) != [
        len(state_list), len(action_list_A), len(action_list_B), len(state_list)
    ]:
        raise Exception('The joint transition tensor for this MDP must have shape [{0}, {1}, {2}, {0}].'.format(
            len(state_list), len(action_list_A), len(action_list_B)
        ))
    
    for state_tensor in joint_transition_tensor:
        for action_tensor_A in state_tensor:
            for action_tensor_B in action_tensor_A:
                if (not (action_tensor_B == 0).all()) and (action_tensor_B.sum() - 1).abs() > tolerance:
                    raise Exception(
                        'Every inner row of the joint transition tensor must either be all zeros '\
                        '(if the action pair can\'t be taken) or sum to 1 '\
                        '(the total probability of ending up in any downstream state).'
                    )

def check_joint_mdp_graph_by_key(joint_mdp_key, tolerance=PROBABILITY_TOLERANCE, mdps_folder=data.MDPS_FOLDER):
    check_joint_mdp_graph(
        data.load_graph_from_dot_file(joint_mdp_key, folder=mdps_folder), tolerance=tolerance
    )

def check_policy_graph(policy_graph, tolerance=PROBABILITY_TOLERANCE):
    policy_tensor = graph.graph_to_policy_tensor(policy_graph)
    state_list, action_list = graph.get_states_from_graph(policy_graph), graph.get_actions_from_graph(policy_graph)

    if list(policy_tensor.shape) != [len(state_list), len(action_list)]:
        raise Exception('Policy tensor must have shape [{0}, {1}].'.format(len(state_list), len(action_list)))
    
    for state_tensor in policy_tensor:
        if (state_tensor.sum() - 1).abs() > tolerance:
            raise Exception(
                'Every row of the policy tensor must sum to 1 ' \
                '(the total probability of taking any action from that state).'
            )

def check_policy_graph_by_key(policy_key, tolerance=PROBABILITY_TOLERANCE, policy_folder=data.POLICIES_FOLDER):
    check_policy_graph(data.load_graph_from_dot_file(policy_key, folder=policy_folder), tolerance=tolerance)

def noop(_):
    pass

def check_discount_rate(discount_rate):
    if discount_rate < 0 or discount_rate > 1:
        raise Exception('The discount rate should be between 0 and 1, not {}.'.format(discount_rate))

def check_num_workers(num_workers):
    if num_workers > mps.cpu_count():
        raise Exception(
            'You can\'t assign more than {} workers on this machine.'.format(mps.cpu_count())
        )

def check_single_state_reward_config(single_state_reward_config, distribution_dict=dist.DISTRIBUTION_DICT):
    check_dict_contains_key(single_state_reward_config, 'dist_name', 'single-state reward config')
    check_dict_contains_key(single_state_reward_config, 'params', 'single-state reward config')

    if single_state_reward_config['dist_name'] not in distribution_dict:
        raise Exception('Undefined distribution \'{}\'.'.format(single_state_reward_config['dist_name']))
    
    if not isinstance(single_state_reward_config['params'], list):
        raise Exception(
            'The \'params\' value of a reward config should be a list, not {}.'.format(
                single_state_reward_config['params']
            )
        )

def check_reward_distribution_config(reward_distribution_config, distribution_dict=dist.DISTRIBUTION_DICT):
    check_dict_contains_key(reward_distribution_config, 'default_dist', 'reward distribution config')
    check_dict_contains_key(reward_distribution_config, 'allow_all_equal_rewards', 'reward distribution config')
    
    check_single_state_reward_config(reward_distribution_config['default_dist'], distribution_dict=distribution_dict)

    for state in reward_distribution_config.get('state_dists', {}).keys():
        check_single_state_reward_config(
            reward_distribution_config['state_dists'][state], distribution_dict=distribution_dict
        )

def check_tensor_or_number_args(args):
    for arg in args:
        if not (torch.is_tensor(arg) or type(arg) == int or type(arg) == float):
            raise Exception('Positional arguments to this function can only be tensors or numbers, not {}.'.format(arg))

# Ensures that each param conforms to a modified version of https://docs.wandb.ai/guides/sweeps/configuration
# along with param-specific sanity checks. In the case of params that are primitive types (str, float,
# list, etc.), the param is directly present in the config. In the case of distributions, we use a dictionary
# in dist.py to recover the actual distribution value. And in the case of MDP or policy graphs, we use dot files
# in the `mdps` or `policies` folders, respectively.
def check_sweep_param(param_name, value_dict, checker_function):
    if set(value_dict.keys()) == set(['value']):
        checker_function(value_dict['value'])
    
    elif set(value_dict.keys()) == set(['values', 'names']):
        if not isinstance(value_dict['values'], list):
            raise Exception('The \'values\' entry for the {} parameter must be a list.'.format(param_name))
        if not isinstance(value_dict['names'], list):
            raise Exception('The \'names\' entry for the {} parameter must be a list.'.format(param_name))
        if len(value_dict['values']) != len(value_dict['names']):
            raise Exception(
                'The \'values\' and \'names\' lists for the {} paramater must have the same length.'.format(
                    param_name
                )
            )
        if not all(isinstance(name, str) for name in value_dict['names']):
            raise Exception(
                'Each item in the \'names\' list of the {} parameter must be a str.'.format(param_name)
            )
        
        for value in value_dict['values']:
            checker_function(value)
    
    else:
        raise Exception(
            'The {} parameter must be set as either {{ \'value\': <value> }} or ' \
            '{{ \'values\': <list_of_values>, \'names\': <list_of_names> }}.'.format(param_name)
        )

def check_sweep_params(sweep_params):
    all_param_checkers = {
        'mdp_graph': check_mdp_graph_by_key,
        'joint_mdp_graph': check_joint_mdp_graph_by_key,
        'policy_graph_agent_B': check_policy_graph_by_key,
        'seed_policy_graph_agent_B': check_policy_graph_by_key,
        'reward_correlation': noop,
        'discount_rate': check_discount_rate,
        'discount_rate_agent_B': check_discount_rate,
        'reward_distribution': check_reward_distribution_config,
        'num_reward_samples': noop,
        'convergence_threshold': noop,
        'value_initializations': noop,
        'num_workers': check_num_workers,
        'random_seed': noop
    }

    for param_name, value_dict in sweep_params.items():
        check_sweep_param(param_name, value_dict, all_param_checkers[param_name]) 
