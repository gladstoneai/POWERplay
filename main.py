from src import test
from src import launch
from src import data

# TODO: Test simple 3-cell gridworld with 2 agents and a fixed policy.
# TODO: Add policy construction and checking methods onto an existing gridworld. Test & document.
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

# TODO NEXT: Refactor gridworld coordinates & functions that go from '(0, 1)' to [0, 1] and vice versa.
# TODO NEXT: Test setting random seed.
# - For every part of the code that requires a random number generator.
# - Ideally you pass in a random number generator for each component.
# TODO NEXT: Run annealing simulations with stochastic transition functions.
# - Slight perturbation: add some noise to all actions and you should see POWER go down everywhere.
#     - Can do annealing: stochasticity increases as you uniformly add noise to all states.
#     - The delta of POWER between states.
# - Wind, teleporters, irreversible actions / doorways.
# TODO: Run scaling tests to confirm we can do ~100 state MDPs in stochastic mode.

if __name__ == '__main__':
    pass

    # test.test_vanilla()
    # test.test_gridworld()
    # test.test_stochastic()
