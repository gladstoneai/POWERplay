from src.lib import data
from src.lib import get
from src import test
from src import launch

# TODO: Add visualization of the MDP graph for the integrated transition tensor.
# TODO: Add a run.py file to implement higher-level workflows for MDP and policy graph construction.
# TODO: Check why beep_when_done doesn't work.
# TODO: Update get_sweep_state_list to handle the multiagent case and return all the MDPs and policies
# in that case.
# TODO: Add sanity checks to policy / MDP inputs for the multiagent case. In particular, add checks that both
# MDPs and the policy have the same state and action sets.
# TODO: Create gridworld visualizations for the multiagent case.
# TODO: Convert the reduced agent A tensor to a standard stochastic MDP graph for visualization / debugging.
# - Make sure this is what gets visualized in the experiment pipeline.
# TODO: Check compatibility of policy and tensor (for multiagent) up front in launch.py param checker.
# TODO: More stochastic experiments.
# - Wind, teleporters, irreversible actions / doorways.
# TODO: Refactor codebase to keep all files under 200 lines each.
# TODO: Add visualization for gridworld and Agent 2 policy.
# TODO: Add method to save (gridworld, policy, MDP) triple.
# TODO: Do profiling and see which parts are the slowest.
# TODO: Experiment with multiple fixed policies in the 3-cell 2-agent gridworld.
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

# compute_optimal_policy_tensor() computes the optimal policy for Agent A. We can use this over
#   multiagent MDPs (i.e., MDPs over which the positions of Agents A and B are explicitly defined).
# policy_tensor_to_graph() converts a policy tensor to a policy graph.

# TODO NEXT:
# - What happens if you correlate Agent B's policy and make it dependent on Agent A's position
# -- For example, have Agent B follow Agent A
# -- Forbid the agents from being in the same square
# -- Try using different reward function distributions like a Jeffries prior
#     (a fat tail prior might give a stronger incentive to fight; the real world has sparser reward than uniform)
# --- You can try this in a single agent setting too
# - Remove code to handle correlation plots for multiagent in render
# - Create a new file for high-level workflows
# - Split viz into viz for graphs and viz for plots
# - Clean up render_gridworld_rollout_snapshot()
# TODO NEXT: Have an animation of what the policy for Agent B does. (Visualizing joint rollouts between A and B.)
# - Improve visualization for rollouts
# -- Show heatmap in the rollout at each timestep as agent B moves
# TODO NEXT: Start building correlated reward functions

if __name__ == '__main__':
    # test.test_vanilla()
    # test.test_gridworld()
    # test.test_stochastic()
    # test.test_multiagent()

    pass



    