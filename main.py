from src.lib.utils import misc
from src.lib import data
from src import test
from src import launch
from src import base

# TODO Engineering nice to haves:
# - Remove code to handle correlation plots for multiagent in render
# - Split viz into viz for graphs and viz for plots
# - Clean up render_gridworld_rollout_snapshot()
# - Add sanity checks to policy / MDP inputs for the multiagent case. In particular, add checks that both
# MDPs and the policy have the same state and action sets.
# - Refactor codebase to keep all files under 200 lines each.
# - Do profiling and see which parts are the slowest.
# - Save with wandb instead of homespun data methods.
# - Investigate writing type hints.
# - Add argparse.ArgumentParser().
# parser = argparse.ArgumentParser()
# parser.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',
#                      help='input batch size for training (default: 8)')
# args = parser.parse_args()
# - Refactor experiment wrapper with https://hydra.cc/ when I understand what experiment configs
# I commonly use.

# TODO Experiment nice to haves:
# - Plot 10th and 90th percentile POWER scores in alignment curves
# - Build arbitrary back-and-forth counter-optimizations between Agent A and Agent B

# TODO Write up:
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

# TODO NEXT:
# - Make Agent A and Agent B move jointly, instead of in turns
# -- Update transition probability calculation for rollouts: https://drive.google.com/file/d/1WD1Rd4K_71sgqg_P8Tp5QMnWk_Sdq5xL/view?usp=sharing
# -- Update code for rollout visualization of joint transitions
# --- Switch checks from check_full_graph_compatibility to check_policy_and_joint_mdp_compatibility
# --- Change graph_to_joint_transition_tensor to graphs_to_multiagent_transition_tensor and delete existing function by that name
# --- Delete compute_multiagent_transition_tensor and replace with compute_full_multiagent_transition_tensor
# --- Delete graphs_to_multiagent_transition_tensor and replace with graphs_to_full_multiagent_transition_tensor in graph
# -- Do a full test of the refactored code
# -- Update README to handle joint multiagent graphs
# - Refactor so that the get_transition_graphs check is only applied once at the beginning, since it slows down graph loading
# - Allow us to plot joint MDP conditional on other agent policy (to avoid huge size of image for joint MDPs)
# - Test how gamma values affect alignment curves
# - What about a multiagent setting that has a teleporter cell? Do you get stronger IC from that?
# - What about a bigger world? This allows us to have a more skewed Bernoulli distribution without numerical instability

if __name__ == '__main__':
    # test.test_vanilla()
    # test.test_gridworld()
    # test.test_stochastic()
    # test.test_multiagent()
    # test.test_reward_correlation()

    pass