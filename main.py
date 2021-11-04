import lib
import data

def test():
    return lib.run_one_experiment(
        adjacency_matrix=data.ADJACENCY_MATRIX,
        state_list=data.STATE_LIST,
        discount_rate=0.9,
        save_experiment=False,
        plot_when_done=True,
        num_workers=1
    )

# TODO: Use PyTorch to create a sampler for reward function distributions.
# (PyTorch has a .sample that you can use to sample from various distributions of reward functions.)
# TODO: Refactor experiment wrapper with https://hydra.cc/
# TODO: Use https://wandb.ai/ for experiment tracking, will log everything in the cloud.
# TODO: Begin first experiment series.

# TODO: Investigate writing type hints.
# TODO: Use networkx to construct graph rather than manually writing the adjacency matrix.
# Refactor this when we start investigating new topologies.

ex = test()