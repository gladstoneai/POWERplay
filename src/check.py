import torch
import pathos.multiprocessing as mps

from . import data
from . import utils

def check_project_exists(project, entity, wb_api):
    if project not in [proj.name for proj in wb_api.projects(entity=entity)]:
        raise Exception(
            'The project {0} doesn\'t exist in the entity {1}. '\
            'You need to create a project of that name before running this sweep.'.format(project, entity)
        )

def check_num_samples(num_samples, num_workers):
    if num_samples % num_workers != 0:
        raise Exception('The number of reward samples must be an exact multiple of the number of workers.')

def check_mdp_graph(mdp_key, mdps_folder=data.MDPS_FOLDER):
    mdp_graph = data.load_graph_from_dot_file(mdp_key, folder=mdps_folder)

    state_action_matrix = utils.graph_to_state_action_matrix(mdp_graph)

    if torch.tensor([(state_action_matrix[i] == 0).all() for i in range(len(state_action_matrix))]).any():
        raise Exception(
            'You can\'t have a row of your adjacency matrix whose entries are all zero, '\
            'because this corresponds to a state with no successor states. Either give that state a self-loop, '\
            'or connect it to a terminal state whose reward is fixed at 0.'
        )

def check_policy(policy, tolerance=1e-4):
    if policy.shape[0] != policy.shape[1]:
        raise Exception('The policy tensor must be n x n.')
    
    if (torch.abs(torch.sum(policy, 1) - 1) > tolerance).any():
        raise Exception('Each row of the policy tensor must sum to 1.')

def noop(param_value):
    pass

def check_discount_rate(discount_rate):
    if discount_rate < 0 or discount_rate > 1:
        raise Exception('The discount rate should be between 0 and 1.')

def check_num_workers(num_workers):
    if num_workers > mps.cpu_count():
        raise Exception(
            'You can\'t assign more than {} workers on this machine.'.format(mps.cpu_count())
        )

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
        'discount_rate': check_discount_rate,
        'reward_distribution': noop,
        'num_reward_samples': noop,
        'convergence_threshold': noop,
        'value_initializations': noop,
        'num_workers': check_num_workers,
        'random_seed': noop
    }

    for param_name, value_dict in sweep_params.items():
        check_sweep_param(param_name, value_dict, all_param_checkers[param_name]) 
