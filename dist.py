import torch

import data

def generate_random_policy(number_of_states, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    raw_policy = torch.rand(number_of_states, number_of_states)
# Each row of the policy sums to 1, since it represents the probabilities of actions from that state.
    return raw_policy / torch.sum(raw_policy, 1, keepdims=True)

def preset_pdf_constructor(distribution_name, *params, distribution_dict=data.DISTRIBUTION_DICT):
    raw_distribution = distribution_dict[distribution_name](*params)
    # Handle both PyTorch and manual functions built with pdf_sampler_constructor
    return raw_distribution if not hasattr(raw_distribution, 'sample') else raw_distribution.sample

def reward_distribution_constructor(
    state_list,
    default_reward_sampler=preset_pdf_constructor('uniform', 0., 1.),
    state_reward_samplers={}
):
    def reward_distribution(number_of_samples):
        state_sampler_list = [(state_reward_samplers[state_list[i]] if (
            state_list[i] in state_reward_samplers.keys()
        ) else default_reward_sampler) for i in range(len(state_list) - 1)]

        return torch.cat([
            sample(torch.tensor([number_of_samples])).unsqueeze(1) for sample in state_sampler_list
        ] + [torch.zeros(number_of_samples, 1)], dim=1)
    
    return reward_distribution