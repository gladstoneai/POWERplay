import lib
import data
import torch.distributions as td
import utils

def test():
    return lib.run_one_experiment(
        adjacency_matrix=data.ADJACENCY_MATRIX,
        state_list=data.STATE_LIST,
        discount_rate=0.9,
        reward_distribution=utils.reward_distribution_constructor(
            data.STATE_LIST,
            default_distribution=utils.ArbitraryRewardDistribution(
                pdf=lambda x: x, interval=(0, 1), resolution=100
            )
        ),
        save_experiment=False,
        plot_when_done=True,
        num_workers=10,
        save_figs=True,
        num_reward_samples=10000
    )

# TODO: Refactor experiment wrapper with https://hydra.cc/
# TODO: Use https://wandb.ai/ for experiment tracking, will log everything in the cloud.
# TODO: Begin first experiment series.

# TODO: Investigate writing type hints.
# TODO: Use networkx to construct graph rather than manually writing the adjacency matrix.
# Refactor this when we start investigating new topologies.


# TODO: Rewrite distribution and experiment as classes, subclassing from Immutable.

