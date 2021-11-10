import torch
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
            default_reward_sampler=td.Uniform(torch.tensor([0.]), torch.tensor([1.])).sample
        ),
        save_experiment_local=True,
        save_experiment_wandb=False,
        experiment_handle='gamma_0p9-dist_unif0t1_iid-samples_10k',
        wandb_run_params={
            'project': 'test-project',
            'notes': 'These are some notes on this project.',
            'mode': 'online'
        },
        plot_when_done=False,
        num_workers=10,
        save_figs=True,
        num_reward_samples=10000
    )

# TODO: Begin first experiment series.

# TODO: Add better tracking of reward distribution for reproducibility.
# TODO: Refactor experiment code so that config is an object (or dict) you pass into it, as opposed to
# a big list of kwargs.
# TODO: Investigate writing type hints.
# TODO: Use networkx to construct graph rather than manually writing the adjacency matrix.
# Refactor this when we start investigating new topologies.
# TODO: Add argparse.ArgumentParser().
# parser = argparse.ArgumentParser()
# parser.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',
#                      help='input batch size for training (default: 8)')
# args = parser.parse_args()
# TODO: Go through wandb API and add all the other stuff I need.
# TODO: Refactor experiment wrapper with https://hydra.cc/ when I understand what experiment configs
# I commonly use.

import viz

experiment = test()