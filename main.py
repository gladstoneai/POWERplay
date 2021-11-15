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
            default_reward_sampler=td.Uniform(torch.tensor([-1.]), torch.tensor([0.])).sample
        ),
        save_experiment_local=True,
        save_experiment_wandb=True,
        experiment_handle='gamma_0p9-dist_unif_n1t0_iid-samples_100k',
        wandb_run_params={
            'project': 'power-project',
            'entity': 'power-experiments',
            'notes': 'Uniform iid distribution of rewards over [-1, 0).'
        },
        plot_when_done=False,
        num_workers=10,
        save_figs=True,
        num_reward_samples=100000
    )

# TODO: Begin first experiment series.

# TODO: Investigate writing type hints.
# TODO: Use networkx to construct graph rather than manually writing the adjacency matrix.
# Refactor this when we start investigating new topologies.
# TODO: Add argparse.ArgumentParser().
# parser = argparse.ArgumentParser()
# parser.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',
#                      help='input batch size for training (default: 8)')
# args = parser.parse_args()
# TODO: Refactor experiment code so that config is an object (or dict) you pass into it, as opposed to
# a big list of kwargs.
# TODO: Go through wandb API and add all the other stuff I need.
# TODO: Refactor experiment wrapper with https://hydra.cc/ when I understand what experiment configs
# I commonly use.

import viz
import pathlib as path

experiment = test()
'''
expt = data.load_experiment('20211114110144-gamma_0p1-dist_unif0t1_iid-samples_100k')
reward_samples, power_samples = expt['outputs']['reward_samples'], expt['outputs']['power_samples']

viz.render_all_outputs(
    reward_samples,
    power_samples,
    data.STATE_LIST,
    sample_filter=reward_samples[:, 3] > reward_samples[:, 4],
    show=False,
    save_fig=True,
    save_handle='FILTER_reward_ℓ_↖_gt_ℓ_↙',
    save_folder=path.Path()/data.EXPERIMENT_FOLDER/expt['name']
)
'''