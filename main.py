import launch
import data

def test():
    launch.launch_sweep(
        'test_sweep.yaml',
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        project='uncategorized'
    )

# TODO: Test setting random seed.
# TODO: Remove privileged status of TERMINAL state (and delete MDP requirements for a TERMINAL state).
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

# TODO NEXT: Change figure rendering so parameter values appear in the title (which makes animations clearer).
# TODO NEXT: Test a gridworld that's two squares with one cell in between (a "bridge").

'''
launch.launch_sweep(
    'sweep-REWARD_ANNEALING_CONJUGATE_BETA_ON_GRIDWORLD-gamma_0p5-samples_100k.yaml',
    entity=data.get_settings_value('public.WANDB_COLLAB_ENTITY'),
    project='power-project',
    plot_as_gridworld=True,
    beep_when_done=True
)
'''