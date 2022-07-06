from src.lib import data
from src import test
from src import launch
from src import base

# TODO: Refactor the loop in the torch.stack line of run_one_experiment (we should be doing it all in PyTorch, not a Python loop)
# TODO: Update get_sweep_state_list to handle the multiagent case and return all the MDPs and policies
# in that case.
# TODO: Add sanity checks to policy / MDP inputs for the multiagent case. In particular, add checks that both
# MDPs and the policy have the same state and action sets.
# TODO: More stochastic experiments.
# - Wind, teleporters, irreversible actions / doorways.
# TODO: Refactor codebase to keep all files under 200 lines each.
# TODO: Add visualization for gridworld and Agent 2 policy.
# TODO: Do profiling and see which parts are the slowest.
# TODO: (?) Create a state container for training loop with stateless functions underneath.
# TODO: Save with wandb instead of homespun data methods.
# TODO: Investigate writing type hints.
# TODO: Add argparse.ArgumentParser().
# parser = argparse.ArgumentParser()
# parser.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',
#                      help='input batch size for training (default: 8)')
# args = parser.parse_args()
# TODO: Refactor experiment wrapper with https://hydra.cc/ when I understand what experiment configs
# I commonly use.
# TODO: Add ability to run sweep without wandb server access (i.e., offline mode). May be impossible, but
# would be great as it would allow me to run local tests without consuming bandwidth, etc.
# TODO: Add sparse / non-sparse flag to toggle value iteration calculation.
# - "Using a sparse storage format for storing sparse arrays can be advantageous only when the
#   size and sparsity levels of arrays are high. Otherwise, for small-sized or low-sparsity arrays
#   using the contiguous memory storage format is likely the most efficient approach."
#   Source: https://pytorch.org/docs/stable/sparse.html

# TODO NEXT:
# - Split up run_one_experiment() into 3 functions for the 3 different sweep types
# - Refactor rewards_to_outputs() and output_sample_calculator() so we don't have None arguments all over the place
# - Maybe also different discount rates for Agent A vs Agent B.
# - Visualizations for Agent B "POWER" proxy? i.e., rather than optimal value averaged over rewards,
#   just look at the value Agent B can get (given A's optimal counter policy), averaged over rewards
# - Final stage: building arbitrary back-and-forth counter-optimizations?
# - Maybe log POWER at each step in a trajectory in a rollout, averaging over trajectories

# - What happens if you correlate Agent B's policy and make it dependent on Agent A's position
# -- Forbid the agents from being in the same square
# - Remove code to handle correlation plots for multiagent in render
# - Split viz into viz for graphs and viz for plots
# - Clean up render_gridworld_rollout_snapshot()
# TODO NEXT: Start building correlated reward functions
# - Go back and forth: do 1 iteration of value iteration for B, 1 for A, etc. and stop at some predefined number
# - Give B more or fewer iterations than A
# - Start misaligning the reward functions; at what point does A "prefer" a less capable B vs a more capable B
# -- HYPOTHESIS: you don't need to be that misaligned to get adversarial behavior
# -- Maybe test out different initializations for Agent A policy (other than just random uniform), maybe sample a random policy for each reward sample

if __name__ == '__main__':
    # test.test_vanilla()
    # test.test_gridworld()
    # test.test_stochastic()
    # test.test_multiagent()
    # test.test_reward_correlation()

    pass

    