import launch
import data

def test():
    launch.launch_sweep(
        'test_sweep.yaml',
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        project='uncategorized'
    )

# TODO: Remove privileged status of TERMINAL state (and delete MDP requirements for a TERMINAL state).
# TODO: Test setting random seed.
# TODO: Put code files in src directory.
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

# TODO NEXT: Plot the POWER numbers for the two-room gridworld. Shouldn't the middle tile have a higher POWER?
# TODO NEXT: Do a run for a 100-cell gridworld at 0.8 discount rate, 5k-10k samples to confirm scaling.
# TODO NEXT: Look at how the POWER of the states changes as we anneal from maxent to a distribution over one-hot
# reward functions. (i.e., the distribution that places equal weight on any one state having a reward of 1, with
# the rest having reward of 0.)
