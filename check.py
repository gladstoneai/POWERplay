import torch
import pathos.multiprocessing as mps

def check_adjacency_matrix(adjacency_matrix):
    if (not (adjacency_matrix[-1][:-1] == 0).all()) or (not adjacency_matrix[-1][-1] == 1):
        raise Exception(
            'The last row of the adjacency matrix {} must be 1 in the last entry, 0 elsewhere. '\
            'The last entry state the terminal state, which can only lead to itself.'.format(adjacency_matrix)
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
# along with param-specific sanity checks. In the case of params that are not primitive types (str, float,
# list, etc.) we use a dictionary in data.py to recover the value from the primitive input param.
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
        'adjacency_matrix': noop,
        'state_list': noop,
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

def check_experiment_inputs(
    adjacency_matrix=None,
    discount_rate=None,
    save_experiment_wandb=None,
    wandb_run_params=None,
    reward_distribution=None,
    state_list=None,
    plot_when_done=None,
    save_figs=None,
    num_workers=None
):
    if (plot_when_done or save_figs) and state_list is None:
        raise Exception('If plotting or saving figures, you need to input the state_list.')

    if num_workers > mps.cpu_count():
        raise Exception(
            'You can\'t assign more than {} workers on this machine.'.format(mps.cpu_count())
        )
    
    if save_experiment_wandb:
        if wandb_run_params.get('project') is None:
            raise Exception('You need to indicate which W&B project this run belongs to.')

    if (not (adjacency_matrix[-1][:-1] == 0).all()) or (not adjacency_matrix[-1][-1] == 1):
        raise Exception(
            'The last row of the adjacency matrix must be 1 in the last entry, 0 elsewhere. '\
            'The last entry represents the terminal state, which can only lead to itself.'
        )
    
    if discount_rate < 0 or discount_rate > 1:
        raise Exception('The discount rate should be between 0 and 1.')
    
    if not callable(reward_distribution):
        raise Exception('The reward_distribution must be a callable function.')