import launch
import data
import dist
import check

def test(reward_inputs, experiment_handle, wandb_notes):
    return launch.run_one_experiment_OLD(
        adjacency_matrix=data.ADJACENCY_MATRIX_DICT['mdp_from_paper'],
        state_list=data.STATE_LIST,
        discount_rate=0.9,
        reward_distribution=dist.reward_distribution_constructor(
            data.STATE_LIST,
            default_reward_sampler=dist.config_to_pdf_constructor(reward_inputs)
        ),
        save_experiment_local=True,
        save_experiment_wandb=False,
        experiment_handle=experiment_handle,
        wandb_run_params={
            'project': 'power-project',
            'entity': 'power-experiments',
            'notes': wandb_notes
        },
        plot_when_done=False,
        num_workers=10,
        save_figs=True,
        num_reward_samples=10000
    )

# Ensure all our adjacency matrices pass sanity checks on startup.
# TODO: Save these in dill by default and sanitize on saving rather than on loading.

if __name__ == '__main__':
    for _, adjacency_matrix in data.ADJACENCY_MATRIX_DICT.items():
         check.check_adjacency_matrix(adjacency_matrix)

# TODO: Build state_list into the program itself since we shouldn't assume a list format.
# TODO: Transition from old experiment function to new experiment function.
# TODO: Delete load_experiment function.
# TODO: Document API for the sweep function.
# TODO: Test setting random seed.
# TODO: Create src directory and update filepaths.
# TODO: Save with wandb instead of homespun data methods.
# TODO: Investigate writing type hints.
# TODO: Use networkx to construct graph rather than manually writing the adjacency matrix.
# Refactor this when we start investigating new topologies.
# TODO: Add argparse.ArgumentParser().
# parser = argparse.ArgumentParser()
# parser.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',
#                      help='input batch size for training (default: 8)')
# args = parser.parse_args()
# TODO: Refactor experiment wrapper with https://hydra.cc/ when I understand what experiment configs
# I commonly use.

# TODO: Add ability to run sweep without wandb server access (i.e., offline mode). May be impossible, but
# would be great as it would allow me to run local tests without consuming bandwidth, etc.

'''
expt = test(
    { 'dist_name': 'uniform', 'params': [0., 2.] },
    'gamma_0p9-dist_unif_0t2_iid-samples_100k',
    'Uniform iid distribution of rewards over [0, 2).'
)
'''
launch.launch_sweep('test_sweep.yaml')
