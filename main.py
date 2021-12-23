import launch
import data

def test():
    launch.launch_sweep(
        'test_sweep.yaml',
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        project='uncategorized'
    )

# TODO: POWER mean graph can be a heatmap of the MDP in the gridworld case.
# TODO: Test a gridworld that's two squares with one cell in between (a "bridge").
# TODO: Change figure rendering so parameter values appear in the title (which makes animations clearer).
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

# TODO NEXT: Improve visualization for correlation plots of gridworlds, so that the squares are in their right positions.
# TODO: Use Bayesian updating & conjugate priors for sweep annealing. (Beta & Dirichlet distributions.)
