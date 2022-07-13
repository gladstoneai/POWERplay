import torch
import functools as func
import operator as op

################################################################################

TINY_NUMBER = 1e-45 # To avoid both divide by 0 and overflow in torch

################################################################################

def build_sweep_config(
    input_sweep_config=None,
    sweep_name=None,
    project=None,
    entity=None
):
    return {
        **input_sweep_config,
        'parameters': { # Allows us to name individual runs with preset strings
            key: (value_dict if ('value' in input_sweep_config.get('parameters').get(key)) else { 'values': [
                [val, name] for val, name in zip(value_dict.get('values'), value_dict.get('names'))
            ]}) for (key, value_dict) in input_sweep_config.get('parameters').items()
        },
        'name': sweep_name,
        'project': project,
        'entity': entity,
        'method': 'grid'
    }

def retrieve(dictionary, path_with_dots):
    return func.reduce(op.getitem, path_with_dots.split('.'), dictionary)

# Biggest fractional change in state value over one iteration.
def calculate_value_convergence(old_values, new_values, tiny_number=TINY_NUMBER):
    return torch.max(torch.abs((old_values - new_values) / (old_values + tiny_number)))

def pdf_sampler_constructor(pdf=lambda x: 1, interval=(0, 1), resolution=100):

# A bit weird, but lets us match the currying signature of PyTorch distribution functions.
    def constructor_l2():

        def pdf_sampler(number_of_samples):
            source_samples = torch.rand(number_of_samples)

            dest_boundary_points = torch.linspace(*(interval + (resolution,)))
            sample_widths = dest_boundary_points[1:] - dest_boundary_points[:-1]
            pdf_at_midpoints = torch.ones(resolution - 1) * pdf(
                0.5 * (dest_boundary_points[1:] + dest_boundary_points[:-1])
            )
            probability_masses = sample_widths * pdf_at_midpoints / torch.sum(sample_widths * pdf_at_midpoints)

            source_boundary_points = torch.cat((torch.zeros(1), torch.cumsum(probability_masses, 0)))
            sample_bins = torch.bucketize(source_samples, source_boundary_points)
            
            sample_bin_positions = (
                source_samples - source_boundary_points[sample_bins - 1]
            ) / (
                source_boundary_points[sample_bins] - source_boundary_points[sample_bins - 1]
            )

            return sample_bin_positions * (
                dest_boundary_points[sample_bins] - dest_boundary_points[sample_bins - 1]
            ) + dest_boundary_points[sample_bins - 1]
        
        return pdf_sampler
    
    return constructor_l2

def build_run_name(local_sweep_name, run_config, sweep_variable_params):
    return '-'.join([local_sweep_name] + [ # sorted() ensures naming is always consistent
        '{0}__{1}'.format(key, run_config[key][1]) for key in sorted(run_config.keys()) if (
            key in sweep_variable_params
        )
    ])

def get_variable_params(sweep_config):
    return [
        param for param in sweep_config.get('parameters').keys() if (
            sweep_config.get('parameters').get(param).get('values') is not None
        )
    ]

def clone_run_inputs(runs_data_with_transition_tensor, ignore_terminal_state=False):
    return {
        run_data['name']: {
            'args': [
                run_data['outputs']['reward_samples'][:,:-1] if (
                    ignore_terminal_state
                ) else run_data['outputs']['reward_samples'],
                run_data['inputs']['transition_tensor'],
                run_data['inputs']['discount_rate']
            ], 'kwargs': {
                'num_workers': run_data['inputs']['num_workers'],
                'convergence_threshold': run_data['inputs']['convergence_threshold']
            }
         } for run_data in runs_data_with_transition_tensor.values()
    }

def set_global_random_seed(random_seed):
    if random_seed is None:
        torch.seed()
    else:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

def determine_sweep_type(run_params):
    if 'mdp_graph' in run_params:
        return 'single_agent'
    
    elif (
        'mdp_graph_agent_A' in run_params
    ) and (
        'mdp_graph_agent_B' in run_params
    ) and (
        'policy_graph_agent_B' in run_params
    ):
        return 'multiagent_fixed_policy'
    
    elif (
        'mdp_graph_agent_A' in run_params
    ) and (
        'mdp_graph_agent_B' in run_params
    ) and (
        'reward_correlation' in run_params
    ):

        if 'state_dists' in run_params.get('reward_distribution'):
            raise Exception(
                'Correlated reward sweeps don\'t currently support state-specific reward distributions.'
            ) # NOTE: We *could* quite easily support state-specific reward distributions, just not with negatively correlated rewards without a lot more work.

        return 'multiagent_with_reward'
    
    else:
        raise Exception('Unknown sweep type.')
    
def tile_tensor(input_object, number_of_copies):
    input_tensor = input_object if torch.is_tensor(input_object) else torch.tensor(input_object)
    return torch.tile(input_tensor, [number_of_copies, *([1] * len(input_tensor.shape))])