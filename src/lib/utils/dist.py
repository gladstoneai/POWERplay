import torch
import torch.distributions as td

from . import misc

################################################################################

DISTRIBUTION_DICT = {
    'uniform': {
        'distribution': td.Uniform,
        'symmetric_interval': lambda x, y: (x, y)
    },
    'uniform_0_1_manual': {
        'distribution': misc.pdf_sampler_constructor(pdf=lambda x: 1, interval=(0, 1), resolution=100),
        'symmetric_interval': (0, 1)
    },
    'jaynes_prior': {
        'distribution': misc.pdf_sampler_constructor(pdf=lambda x: 1/(x * (1 - x)), interval=(0, 1), resolution=100),
        'symmetric_interval': (0, 1)
    },
    'beta': {
        'distribution': td.Beta
    },
    'pareto': {
        'distribution': td.Pareto
    },
    'bernoulli': {
        'distribution': td.Bernoulli
    }
}

################################################################################

# A 1d pdf config has the form
# { 'dist_name': <key for distribution in DISTRIBUTION_DICT>, 'params': <params input to that distribution> }
def config_to_pdf_constructor(distribution_config, distribution_dict=DISTRIBUTION_DICT):
    raw_distribution = distribution_dict[distribution_config['dist_name']]['distribution'](
        *distribution_config.get('params')
    )
    # Handle both PyTorch and manual functions built with pdf_sampler_constructor
    return raw_distribution if not hasattr(raw_distribution, 'sample') else raw_distribution.sample

def config_to_symmetric_interval(distribution_config, distribution_dict=DISTRIBUTION_DICT):
    distribution_data = distribution_dict[distribution_config['dist_name']]
    dist_params = distribution_config['params']
    interval = distribution_data.get('symmetric_interval')

    if interval is None:
        return None
    
    else:
        # Handle both PyTorch and manual functions built with pdf_sampler_constructor
        return interval if not hasattr(
            distribution_data['distribution'](*dist_params), 'sample'
        ) else interval(*dist_params)

# if allow_all_equal_rewards is False, re-sample anytime you get a reward function for which the rewards
# are equal over all states. This is especially useful for e.g., the Bernoulli distribution (which is
# a good distribution to use when investigating instrumental convergence), because you'll often have
# reward distributions that are all zeros or all ones by chance.
def reward_distribution_constructor(
    state_list,
    default_reward_sampler=config_to_pdf_constructor({ 'dist_name': 'uniform', 'params': [0., 1.] }),
    state_reward_samplers={},
    allow_all_equal_rewards=True
):
    def reward_distribution(number_of_samples):
        state_sampler_list = [(state_reward_samplers[state_list[i]] if (
            state_list[i] in state_reward_samplers.keys()
        ) else default_reward_sampler) for i in range(len(state_list))]

        outputs_ = torch.cat([
            sample(torch.tensor([number_of_samples])).unsqueeze(1) for sample in state_sampler_list
        ], dim=1)

        # If we forbid reward functions whose rewards are equal at all states, our sampler discards all reward samples
        # that fail the test, and it re-runs as many times as we need to get all passing samples.
        if not allow_all_equal_rewards:
            if outputs_.shape[0] == 0:
                return outputs_
            
            else:
                allowed_outputs_ = outputs_[torch.all(outputs_ == outputs_[:,0].unsqueeze(1), dim=1).logical_not()]
                return torch.cat([
                    allowed_outputs_,
                    reward_distribution(number_of_samples - allowed_outputs_.shape[0])
                ], dim=0)

        else:
            return outputs_
    
    return reward_distribution

# A distribution config has the form
# { 'default_dist': <1d pdf config applied iid to all states>, 'state_dists': { 
#   <state label 1>: <1d pdf config applied to state 1 that overrides the default dist>
#   <state label 2>: <1d pdf config applied to state 2 that overrides the default dist>
#   ... etc.
# } }
def config_to_reward_distribution(
    state_list,
    reward_dist_config,
    distribution_dict=DISTRIBUTION_DICT
):
    return reward_distribution_constructor(
        state_list,
        default_reward_sampler=config_to_pdf_constructor(
            reward_dist_config['default_dist'], distribution_dict=distribution_dict
        ),
        state_reward_samplers={
            state: config_to_pdf_constructor(
                reward_dist_config['state_dists'][state], distribution_dict=distribution_dict
            ) for state in reward_dist_config.get('state_dists', {}).keys()
        },
        allow_all_equal_rewards=reward_dist_config.get('allow_all_equal_rewards', True)
    )

def sample_from_state_list(state_list, distribution_vector):
    return state_list[td.Categorical(distribution_vector).sample().item()]

# See https://drive.google.com/file/d/1aCMAainYY_24ihCmjvz-Z_LYr7FLtQ1N/view for calculations that apply
# for correlation coefficients from 0 to 1. Note that similar logic can be used to calculate the correlated rewards
# for correlations from -1 to 0, but negative correlations *only* make sense for pdfs that are symmetric over their
# support.
# single_agent_reward_dist: Output of reward_distribution_constructor(state_list)
# agent_A_samples: Output of reward_distribution_constructor(state_list)(num_samples), a tensor of size
# num_samples x len(state_list)
def generate_correlated_reward_samples(
    single_agent_reward_dist,
    agent_A_samples,
    correlation=1,
    symmetric_interval=None
):
    
    if correlation >= 0 and correlation <= 1:
        prob_mask = td.Categorical(torch.tensor([1 - correlation, correlation])).sample(agent_A_samples.shape)
        return prob_mask * agent_A_samples + (1 - prob_mask) * single_agent_reward_dist(agent_A_samples.shape[0]) # single_agent_reward_dist already knows how many states our MDP has
    
    elif correlation < 0 and correlation >= -1:
        if symmetric_interval is None:
            raise Exception('Negative correlations only work correctly with symmetric reward distributions.')

        prob_mask = td.Categorical(torch.tensor([1 + correlation, -correlation])).sample(agent_A_samples.shape)
        return (
            prob_mask * (sum(symmetric_interval) - agent_A_samples) # We remap the Agent A samples and reverse them through the distribution's axis of symmetry
        ) + (
            (1 - prob_mask) * single_agent_reward_dist(agent_A_samples.shape[0])
        )

    else:
        raise Exception('Only correlations between -1 and 1 are supported; your correlation was {}.'.format(correlation))