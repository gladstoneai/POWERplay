from src import test

# TODO: Run annealing simulations with stochastic transition functions.
# TODO: Test simple 3-cell gridworld with 2 agents and a fixed policy.
# TODO: Add policy construction and checking methods onto an existing gridworld. Test & document.
# TODO: Add visualization for gridworld and Agent 2 policy.
# TODO: Add method to save (gridworld, policy, MDP) triple.
# TODO: Experiment with multiple fixed policies in the 3-cell 2-agent gridworld.
# TODO: Test setting random seed.
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

# TODO NEXT? Fix multiprocessing bug that freezes execution above 30k samples.
# TODO NEXT: Use unsqueeze() on the value_function by default in value_iteration in learn.py.
# TODO NEXT? Add sparse / non-sparse flag to toggle value iteration calculation.
# - "Using a sparse storage format for storing sparse arrays can be advantageous only when the
#   size and sparsity levels of arrays are high. Otherwise, for small-sized or low-sparsity arrays
#   using the contiguous memory storage format is likely the most efficient approach."
#   Source: https://pytorch.org/docs/stable/sparse.html
# TODO NEXT: Add checks for transition_tensor and state_action_matrix.
# - If transition_tensor is None, state_action_matrix should be square.
# - transition_tensor should have all zeros in axis 2 in places where state_action_matrix has a zero.
# - transition_tensor should have every axis 2 sum to 1 in places where state_action_matrix has a 1.
# - Dimensionality of transition_tensor should be (states, actions, states).
# - If transition_tensor is not None, state_action_matrix should have dimension (states, actions).
# TODO NEXT: Add graph visualization for MDPs that represent stochastic transition functions.

rewards, powers = test.test_stochastic()