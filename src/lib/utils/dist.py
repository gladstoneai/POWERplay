import torch
import torch.distributions as td

from . import misc

################################################################################

DISTRIBUTION_DICT = {
    'uniform': td.Uniform,
    'uniform_0_1_manual': misc.pdf_sampler_constructor(pdf=lambda x: 1, interval=(0, 1), resolution=100),
    'jaynes_prior': misc.pdf_sampler_constructor(pdf=lambda x: 1/(x * (1 - x)), interval=(0, 1), resolution=100),
    'beta': td.Beta,
    'pareto': td.Pareto
}

################################################################################

# A 1d pdf config has the form
# { 'dist_name': <key for distribution in DISTRIBUTION_DICT>, 'params': <params input to that distribution> }
def config_to_pdf_constructor(distribution_config, distribution_dict=DISTRIBUTION_DICT):
    raw_distribution = distribution_dict[distribution_config.get('dist_name')](*distribution_config.get('params'))
    # Handle both PyTorch and manual functions built with pdf_sampler_constructor
    return raw_distribution if not hasattr(raw_distribution, 'sample') else raw_distribution.sample

def reward_distribution_constructor(
    state_list,
    default_reward_sampler=config_to_pdf_constructor({ 'dist_name': 'uniform', 'params': [0., 1.] }),
    state_reward_samplers={}
):
    def reward_distribution(number_of_samples):
        state_sampler_list = [(state_reward_samplers[state_list[i]] if (
            state_list[i] in state_reward_samplers.keys()
        ) else default_reward_sampler) for i in range(len(state_list))]

        return torch.cat([
            sample(torch.tensor([number_of_samples])).unsqueeze(1) for sample in state_sampler_list
        ], dim=1)
    
    return reward_distribution

# A distribution config has the form
# { 'default_dist': <1d pdf config applied iid to all states>, 'state_dists': { 
#   <state label 1>: <1d pdf config applied to state 1 that overrides the default dist>
#   <state label 2>: <1d pdf config applied to state 2 that overrides the default dist>
#   ... etc.
# } }
def config_to_reward_distribution(state_list, reward_dist_config, distribution_dict=DISTRIBUTION_DICT):
    return reward_distribution_constructor(
        state_list,
        default_reward_sampler=config_to_pdf_constructor(
            reward_dist_config.get('default_dist'), distribution_dict=distribution_dict
        ),
        state_reward_samplers={
                state: config_to_pdf_constructor(
                    reward_dist_config.get('state_dists').get(state), distribution_dict=distribution_dict
                ) for state in reward_dist_config.get('state_dists', {})
            }
    )

def sample_from_state_list(state_list, distribution_vector):
    return state_list[td.Categorical(distribution_vector).sample().item()]

# See https://drive.google.com/file/d/1aCMAainYY_24ihCmjvz-Z_LYr7FLtQ1N/view?usp=sharing for calculations that apply
# for correlation coefficients from 0  to 1. Note that similar logic can be used to calculate the correlated rewards
# for correlations from -1 to 0, but negative correlations *only* make sense for pdfs that are symmetric over their support.
# single_agent_reward_dist: Output of reward_distribution_constructor(state_list)
# agent_A_samples: Output of reward_distribution_constructor(state_list)(num_samples), a tensor
# of size num_samples x len(state_list)
def generate_correlated_reward_samples(single_agent_reward_dist, agent_A_samples, correlation=1):
    
    if correlation >= 0 and correlation <= 1:
        prob_mask = td.Categorical(torch.tensor([1 - correlation, correlation])).sample(agent_A_samples.shape)
        return prob_mask * agent_A_samples + (1 - prob_mask) * single_agent_reward_dist(len(agent_A_samples))
    
    else:
        raise Exception('Only correlations between 0 and 1 are supported; your correlation was {}.'.format(correlation))