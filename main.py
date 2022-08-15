from src.lib.utils import misc
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
# - Do a rollout visualization to try to debug the interaction experiment
# - Plot Agent A's non-optimal POWER in the case where it's moving randomly (i.e. A is in fact doing better by optimizing)
# - Create a scatter plot of the POWER means between definition (1) and definition (3) POWERs to see if POWERs at equivalent states are identical
# Overnight experiments:
# - Different gamma values
# - Experiment with non-orthogonal action spaces
# - Run for 2 more iterations as another venue to explore
# - Think about: how do we tell the story of Agent B == humans; Agent A == AI, but consistent with joint states / random initial Agent A policy?
# -- Start articulating concisely: what are the insights from these graphs & charts?
# -- Instead of 2 pages for the multiagent POWER definition, create one that's 0.5-1 page in length
# --- Add an intro:
# --- AI alignment, powerful vs not powerful agent, one agent has goal X, another has goal Y, hey this is starting to look like RL
# --- Draw the analogy we laid out earlier into the formula
# --- Can invert this: describe the main outcome first. Here is the multiagent POWER definition, gloss over
#       the details of what these things are. Gloss over the symbol, just here is the policy for A. Only later
#       describe how we compute the policy for A.
# --- Motivate with a human-like example. Introduce notation: 2 agents, one is stronger than the other. Introduce
#       the definition without going into specific details. A formalism that corresponds to the high level intuition
#       in words.
# --- Then talk more concretely about the details. Assumptions about the initial Agent A policy, here's how we do
#       the optimization, here are the figures that we see; introduce the details later. One might be more compelling
#       than the other.
# --- Create a doc with a high level structure.
# --- Content-wise, this is really strong already.
# - Plot 10th and 90th percentile POWER scores
# - Allow us to not plot reward distributions (it's getting to be too many plots)
# - Investigate whether rollouts before steady state are longer for correlation -1 rather than for correlation 1 rewards
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
# -- Maybe test out different initializations for Agent A policy (other than just random uniform), maybe sample a random policy for each reward sample

if __name__ == '__main__':
    # test.test_vanilla()
    # test.test_gridworld()
    # test.test_stochastic()
    # test.test_multiagent()
    # test.test_reward_correlation()

    pass

    