from src import test
from src import launch
from src import data

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

# TODO NEXT: Have an animation of what the policy for Agent B does. (Visualizing joint rollouts between A and B.)
# - Start with the MDP version, because it's more general and you can see where the rewards are.
# -- Then, build a function that takes the resulting policy tensor and converts it to a policy *graph*, so we can visualize it
# i.e., Complete this function
'''
def policy_tensor_to_graph(policy_tensor, state_list=None, action_list=None):
    output_state_list = [
        i for i in range(policy_tensor.shape[0])
    ] if (state_list is None) else state_list
    output_action_list = [
        i for i in range(policy_tensor.shape[1])
    ] if (action_list is None) else action_list

    '''
# -- Make A the rows and B the columns; they won't always have an orthogonal action set, but when they do, this will make it easier to see who is doing what.
# -- Start by plotting just the state set and not the transitions.
# --- Then maybe plot the transitions as arrows of different colors for the two agents?
# -- Show the reward values for Agent A at each joint state
# -- At each step, you can maybe show just the edges from the current state (i.e., the "possible moves")
# -- Produce a series of plots (not an animation yet) highlighting the current state at each step of the rollout; input should be the number of steps in the rollout
# -- Show which agent has JUST moved (initial state should be marked "initial")
# -- Then create an animation of the full rollout
# - This would be for one reward sample.
# - Can do a gridworld version & an MDP version. (MDP, you can see where the rewards are.)
# - Then do the gridworld version
# -- 
# - Maybe a midlevel version between aggregates and individual runs?

if __name__ == '__main__':
    # test.test_vanilla()
    # test.test_gridworld()
    # test.test_stochastic()
    # test.test_multiagent()

    pass