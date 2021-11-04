import pathos.multiprocessing as mps
import torch

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
    num_reward_samples=None,
    reward_distributions=None,
    reward_ranges=None,
    value_initializations=None,
    reward_sample_resolution=None,
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

    if (not (adjacency_matrix[-1][:-1] == 0).all()) or (not adjacency_matrix[-1][-1] == 1):
        raise Exception(
            'The last row of the adjacency matrix must be 1 in the last entry, 0 elsewhere. '\
            'The last entry represents the terminal state, which can only lead to itself.'
        )
    
    if discount_rate < 0 or discount_rate > 1:
        raise Exception('The discount rate should be between 0 and 1.')
    
    if isinstance(reward_ranges, list):
        if len(reward_ranges) != len(adjacency_matrix) - 1:
            raise Exception(
                'The list of reward_ranges must have one entry less than the number of states, '\
                'since the last state in the adjacency_matrix is always the terminal state.'
            )
        
        for interval in reward_ranges:
            if interval[0] >= interval[1]:
                raise Exception(
                    'Each element of reward_ranges must be a 2-tuple '\
                     'whose second entry is bigger than its first.'
                )
    
    if isinstance(reward_ranges, tuple) and reward_ranges[0] >= reward_ranges[1]:
        raise Exception(
            'The interval reward_ranges must be a 2-tuple whose second entry is bigger than its first.'
        )
    
    if isinstance(reward_distributions, list):
        if len(reward_distributions) != len(adjacency_matrix) - 1:
            raise Exception(
                'The list of reward_distributions must have one entry less than the number of states, '\
                'since the last state in the adjacency_matrix is always the terminal state.'
            )
        
        for pdf, interval in zip(reward_distributions, reward_ranges):
            if (torch.ones(reward_sample_resolution) * pdf(
                torch.linspace(*(interval + (reward_sample_resolution,)))
            ) < 0).any():
                raise Exception(
                    'Every pdf in reward_distributions must be positive '\
                    'everywhere on its support in reward_ranges.'
                )
    
    if callable(reward_distributions) and (
            torch.ones(reward_sample_resolution) * reward_distributions(
                torch.linspace(*(reward_ranges + (reward_sample_resolution,)))
            ) < 0
        ).any():
        raise Exception('The function reward_distributions must be positive everywhere in reward_ranges.')

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

def sample_from_pdf(number_of_samples, pdf=lambda x: 1, interval=(0, 1), resolution=100, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
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

# Random rewards over states, sampled independently over interval according to target_distribution (which
# can be un-normalized).
def generate_random_reward(
    number_of_states,
    target_distributions=lambda x: 1,
    intervals=(0, 1),
    resolution=100,
    seed=None
):
    pdfs_list = target_distributions if isinstance(
        target_distributions, list
    ) else [target_distributions] * (number_of_states - 1)
    intervals_list = intervals if isinstance(intervals, list) else [intervals] * (number_of_states - 1)

# The last state listed is conventionally the terminal state, and always has reward 0.
# (The same must hold for all value initializations.)
    return torch.cat((
        torch.tensor([sample_from_pdf(
            1, pdf=pdf, interval=interval, resolution=resolution, seed=seed
        ) for pdf, interval in zip(pdfs_list, intervals_list)]),
        torch.zeros(1)
    ))