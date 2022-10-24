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

def check_identical_reward_sets(reward_dist_config):
    for identical_reward_set_1 in reward_dist_config.get('states_with_identical_rewards', []):
        for identical_reward_set_2 in reward_dist_config.get('states_with_identical_rewards', []):
            if (
                identical_reward_set_1 != identical_reward_set_2
            ) and not (
                set(identical_reward_set_1).isdisjoint(identical_reward_set_2)
            ):
                raise Exception(
                    'Identical reward sets {0} and {1} must be disjoint.'.format(
                        identical_reward_set_1, identical_reward_set_2
                    )
                )
    
    for identical_reward_set in reward_dist_config.get('states_with_identical_rewards', []):
        if reward_dist_config.get('state_dists') is not None:
            identical_reward_dists = [
                reward_dist_config['state_dists'].get(
                    state, reward_dist_config['default_dist']
                ) for state in identical_reward_set
            ]

            if not all([identical_reward_dists[0] == reward_dist for reward_dist in identical_reward_dists]):
                raise Exception(
                    'States with identical rewards must all have the same reward distribution: {}.'.format(
                        identical_reward_set
                    )
                )

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
# states_with_identical_rewards is a list of lists. For example, if it's [['1', '2'], ['3', '4']],
# then state '1' will have the same reward as state '2', and state '3' will have the same reward as
# state '4'.
def reward_distribution_constructor(
    state_list,
    default_reward_sampler=config_to_pdf_constructor({ 'dist_name': 'uniform', 'params': [0., 1.] }),
    state_reward_samplers={},
    states_with_identical_rewards=[],
    allow_all_equal_rewards=True
):
    def reward_distribution(number_of_samples):
        state_sampler_list = [
            (state_reward_samplers[state_list[i]] if (
                state_list[i] in state_reward_samplers.keys()
            ) else default_reward_sampler) for i in range(len(state_list))
        ]

        outputs_ = torch.cat(
            [
                sample(torch.tensor([number_of_samples])).unsqueeze(1) for sample in state_sampler_list
            ], dim=1
        )

        # Overwrite outputs to force identical_reward_states to return identical rewards
        for identical_reward_set in states_with_identical_rewards:
            identical_state_indices = [state_list.index(state) for state in identical_reward_set]

            for identical_state_index in identical_state_indices[1:]:
                outputs_[:, identical_state_index] = outputs_[:, identical_state_indices[0]]

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

def config_to_reward_distribution(
    state_list,
    reward_dist_config,
    distribution_dict=DISTRIBUTION_DICT
):
    check_identical_reward_sets(reward_dist_config)

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
        states_with_identical_rewards=reward_dist_config.get('states_with_identical_rewards', []),
        allow_all_equal_rewards=reward_dist_config.get('allow_all_equal_rewards', True)
    )

def sample_from_state_list(state_list, distribution_vector):
    return state_list[td.Categorical(distribution_vector).sample().item()]

# See https://drive.google.com/file/d/1aCMAainYY_24ihCmjvz-Z_LYr7FLtQ1N/view for calculations that apply
# to correlation coefficients from 0 to 1. Note that similar logic can be used to calculate the correlated rewards
# for correlations from -1 to 0, but negative correlations *only* make sense for pdfs that are symmetric over their
# support.
def generate_correlated_reward_samples(
    single_agent_reward_dist,
    agent_H_samples,
    correlation=1,
    symmetric_interval=None
):
    
    if correlation >= 0 and correlation <= 1:
        prob_mask = td.Categorical(torch.tensor([1 - correlation, correlation])).sample(agent_H_samples.shape)
        return prob_mask * agent_H_samples + (1 - prob_mask) * single_agent_reward_dist(agent_H_samples.shape[0]) # single_agent_reward_dist already knows how many states our MDP has
    
    elif correlation < 0 and correlation >= -1:
        if symmetric_interval is None:
            raise Exception('Negative correlations only work correctly with symmetric reward distributions.')

        prob_mask = td.Categorical(torch.tensor([1 + correlation, -correlation])).sample(agent_H_samples.shape)
        return (
            prob_mask * (sum(symmetric_interval) - agent_H_samples) # We remap the Agent H samples and reverse them through the distribution's axis of symmetry
        ) + (
            (1 - prob_mask) * single_agent_reward_dist(agent_H_samples.shape[0])
        )

    else:
        raise Exception('Only correlations between -1 and 1 are supported; your correlation was {}.'.format(correlation))