import multiprocessing as mps
import warnings as warn
import torch

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

def check_num_samples(num_samples, num_workers):
    if num_samples % num_workers != 0:
        raise Exception('The number of reward samples must be an exact multiple of the number of workers.')

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

# graph_1 and graph_2 could be policy or MDP graphs
def check_graph_state_compatibility(graph_1, graph_2):
    if graph.get_states_from_graph(graph_1) != graph.get_states_from_graph(graph_2):
        raise Exception('The two input graphs must have the same state set.')

# graph_1 and graph_2 could be policy or MDP graphs
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

def check_stochastic_state_name(name):
    if '__' in name:
        raise Exception(
            'The state or action {} has a double underscore in its name, which is forbidden'.format(name)
        )
    
    if ']' in name or '[' in name:
        raise Exception('The state or action {} can\'t have a [ or ] character in its name.'.format(name))

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

def check_mdp_graph(mdp_key, tolerance=PROBABILITY_TOLERANCE, mdps_folder=data.MDPS_FOLDER):
    mdp_graph = data.load_graph_from_dot_file(mdp_key, folder=mdps_folder)
    transition_tensor = graph.graph_to_transition_tensor(mdp_graph)
    state_list, action_list = graph.get_states_from_graph(mdp_graph), graph.get_actions_from_graph(mdp_graph)

    if list(transition_tensor.shape) != [len(state_list), len(action_list), len(state_list)]:
        raise Exception('The transition tensor for MDP {0} must have shape [{1}, {2}, {1}].'.format(
            mdp_key, len(state_list), len(action_list)
        ))
    
    for state_tensor in transition_tensor:
        for action_tensor in state_tensor:
            if (not (action_tensor == 0).all()) and (action_tensor.sum() - 1).abs() > tolerance:
                raise Exception(
                    'Every inner row of the transition tensor {} must either be all zeros ' \
                    '(if the action can\'t be taken) or sum to 1 ' \
                    '(the total probability of ending up in any downstream state).'.format(mdp_key)
                )

def check_policy_graph(policy_key, tolerance=PROBABILITY_TOLERANCE, policy_folder=data.POLICIES_FOLDER):
    policy_graph = data.load_graph_from_dot_file(policy_key, folder=policy_folder)
    policy_tensor = graph.graph_to_policy_tensor(policy_graph)
    state_list, action_list = graph.get_states_from_graph(policy_graph), graph.get_actions_from_graph(policy_graph)

    if list(policy_tensor.shape) != [len(state_list), len(action_list)]:
        raise Exception('The policy tensor for policy {0} must have shape [{1}, {2}].'.format(
            policy_key, len(state_list), len(action_list)
        ))
    
    for state_tensor in policy_tensor:
        if (state_tensor.sum() - 1).abs() > tolerance:
            raise Exception(
                'Every row of the policy tensor {} must sum to 1' \
                '(the total probability of taking an action from that state.'.format(policy_key)
            )

def noop(_):
    pass

def check_discount_rate(discount_rate):
    if discount_rate < 0 or discount_rate > 1:
        raise Exception('The discount rate should be between 0 and 1.')

def check_num_workers(num_workers):
    if num_workers > mps.cpu_count():
        raise Exception(
            'You can\'t assign more than {} workers on this machine.'.format(mps.cpu_count())
        )

def check_tensor_or_number_args(args):
    for arg in args:
        if not (torch.is_tensor(arg) or type(arg) == int or type(arg) == float):
            raise Exception('Positional arguments to this function can only be tensors or numbers, not {}.'.format(arg))

# Ensures that each param conforms to a modified version of https://docs.wandb.ai/guides/sweeps/configuration
# along with param-specific sanity checks. In the case of params that are primitive types (str, float,
# list, etc.), the param is directly present in the config. In the case of distributions, we use a dictionary
# in data.py to recover the actual distribution value. And in the case of MDP graphs, we use tracked dot files
# in the `mdps` folder.
def check_sweep_param(param_name, value_dict, checker_function):
    if set(value_dict.keys()) == set(['value']):
        checker_function(value_dict.get('value'))
    
    elif set(value_dict.keys()) == set(['values', 'names']):
        if not isinstance(value_dict.get('values'), list):
            raise Exception('The \'values\' entry for the {} parameter must be a list.'.format(param_name))
        if not isinstance(value_dict.get('names'), list):
            raise Exception('The \'names\' entry for the {} parameter must be a list.'.format(param_name))
        if len(value_dict.get('values')) != len(value_dict.get('names')):
            raise Exception(
                'The \'values\' and \'names\' lists for the {} paramater must have the same length.'.format(
                    param_name
                )
            )
        if not all(isinstance(name, str) for name in value_dict.get('names')):
            raise Exception(
                'Each item in the \'names\' list of the {} parameter must be a str.'.format(param_name)
            )
        
        for value in value_dict.get('values'):
            checker_function(value)
    
    else:
        raise Exception(
            'The {} parameter must be set as either {{ \'value\': <value> }} or ' \
            '{{ \'values\': <list_of_values>, \'names\': <list_of_names> }}.'.format(param_name)
        )

def check_sweep_params(sweep_params):
    all_param_checkers = {
        'mdp_graph': check_mdp_graph,
        'mdp_graph_agent_A': check_mdp_graph,
        'mdp_graph_agent_B': check_mdp_graph,
        'policy_graph_agent_B': check_policy_graph,
        'reward_correlation': noop,
        'discount_rate': check_discount_rate,
        'discount_rate_agent_B': check_discount_rate,
        'reward_distribution': noop,
        'num_reward_samples': noop,
        'convergence_threshold': noop,
        'value_initializations': noop,
        'num_workers': check_num_workers,
        'random_seed': noop
    }

    for param_name, value_dict in sweep_params.items():
        check_sweep_param(param_name, value_dict, all_param_checkers[param_name]) 
