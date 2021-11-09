import pathos.multiprocessing as mps
import torch
import torch.distributions as td

################################################################################

NEG_INFTY = -1e309
TINY_NUMBER = 1e-45 # To avoid both divide by 0 and overflow in torch

################################################################################

def check_policy(policy, tolerance=1e-4):
    if policy.shape[0] != policy.shape[1]:
        raise Exception('The policy tensor must be n x n.')
    
    if (torch.abs(torch.sum(policy, 1) - 1) > tolerance).any():
        raise Exception('Each row of the policy tensor must sum to 1.')

def check_experiment_inputs(
    adjacency_matrix=None,
    discount_rate=None,
    save_experiment_wandb=None,
    wandb_run_params=None,
    num_reward_samples=None,
    reward_distribution=None,
    value_initializations=None,
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

    if isinstance(value_initializations, list) and len(value_initializations) != num_reward_samples:
        raise Exception('The list of value_initializations must have num_reward_samples values.')
    
    if isinstance(value_initializations, list):
        for val_init in value_initializations:
            if val_init[-1] != 0:
                raise Exception(
                    'The last entry of a value or reward function must be 0, '\
                    'because this corresponds to the terminal state.'
                )
    
    if isinstance(value_initializations, torch.Tensor):
        if value_initializations[-1] != 0:
            raise Exception(
                    'The last entry of a value or reward function must be 0, '\
                    'because this corresponds to the terminal state.'
                )

# Biggest fractional change in state value over one iteration.
def calculate_value_convergence(old_values, new_values, tiny_number=TINY_NUMBER):
    return torch.max(torch.abs((old_values - new_values) / (old_values + tiny_number)))

def generate_random_policy(number_of_states, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    raw_policy = torch.rand(number_of_states, number_of_states)
# Each row of the policy sums to 1, since it represents the probabilities of actions from that state.
    return raw_policy / torch.sum(raw_policy, 1, keepdims=True)

def pdf_sampler_constructor(pdf=lambda x: 1, interval=(0, 1), resolution=100):

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

        return torch.unsqueeze(sample_bin_positions * (
            dest_boundary_points[sample_bins] - dest_boundary_points[sample_bins - 1]
        ) + dest_boundary_points[sample_bins - 1], 1)
    
    return pdf_sampler

def reward_distribution_constructor(
    state_list,
    default_reward_sampler=td.Uniform(torch.tensor([0.]), torch.tensor([1.])).sample,
    state_reward_samplers={}
):
    def reward_distribution(number_of_samples):
        state_sampler_list = [(state_reward_samplers[state_list[i]] if (
            state_list[i] in state_reward_samplers.keys()
        ) else default_reward_sampler) for i in range(len(state_list) - 1)]

        return torch.cat([
            sample(torch.tensor([number_of_samples])) for sample in state_sampler_list
        ] + [torch.zeros(number_of_samples, 1)], dim=1)
    
    return reward_distribution
