import run
import data
import dist
import check

def test(reward_inputs, experiment_handle, wandb_notes):
    return run.run_one_experiment(
        adjacency_matrix=data.ADJACENCY_MATRIX_DICT['mdp_from_paper'],
        state_list=data.STATE_LIST,
        discount_rate=0.9,
        reward_distribution=dist.reward_distribution_constructor(
            data.STATE_LIST,
            default_reward_sampler=dist.preset_pdf_constructor(*reward_inputs)
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

if __name__ == '__main__':
# Ensure all our adjacency matrices pass sanity checks on startup.
# TODO: Save these in dill by default and sanitize on saving rather than on loading.
    pass # TEMP; RESTORE BELOW
    # for _, adjacency_matrix in data.ADJACENCY_MATRIX_DICT.items():
     #    check.check_adjacency_matrix(adjacency_matrix)

# TODO: See run.py file (end).

# TODO: Test setting random seed.
# TODO: Create src directory and update filepaths.
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
'''
reward_inputs = [
    ['uniform', -4., -2.],
    ['uniform', -3., -1.],
    ['uniform', -2., 0.],
    ['uniform', -1., 1.],
    ['uniform', 0., 2.],
    ['uniform', 1., 3.],
    ['uniform', 2., 4.]
]
experiment_handles = [
    'gamma_0p9-dist_unif_n4tn2_iid-samples_100k',
    'gamma_0p9-dist_unif_n3tn1_iid-samples_100k',
    'gamma_0p9-dist_unif_n2t0_iid-samples_100k',
    'gamma_0p9-dist_unif_n1t1_iid-samples_100k',
    'gamma_0p9-dist_unif_0t2_iid-samples_100k',
    'gamma_0p9-dist_unif_1t3_iid-samples_100k',
    'gamma_0p9-dist_unif_2t4_iid-samples_100k'
]
wandb_notes = [
    'Uniform iid distribution of rewards over [-4, -2).',
    'Uniform iid distribution of rewards over [-3, -1).',
    'Uniform iid distribution of rewards over [-2, 0).',
    'Uniform iid distribution of rewards over [-1, 1).',
    'Uniform iid distribution of rewards over [0, 2).',
    'Uniform iid distribution of rewards over [1, 3).',
    'Uniform iid distribution of rewards over [2, 4).'
]

inputs = [(r, e, w) for r, e, w in zip(reward_inputs, experiment_handles, wandb_notes)]
expts = [test(*(inputs[6]))]
'''

sweep_params = {
    'adjacency_matrix': {
        'value': 'temporary_test_values' ## TEMP 'mdp_from_paper'
    },
    'state_list': {
        'value': data.STATE_LIST
    },
    'discount_rate': {
        'values': [0.1, 0.3, 0.5], # , 0.7, 0.9]
        'names': ['0p1', '0p3', '0p5']
    },
    'reward_distribution': {
        'value': {
            'default_dist': {
                'dist_name': 'uniform',
                'params': [0., 1.]
            }
        }
    },
    'num_reward_samples': {
        'value': 100000
    },
    'convergence_threshold': {
        'value': 1e-4
    },
    'num_workers': {
        'value': 10
    },
    'random_seed': {
        'value': None
    }
}

run.run_experiment_sweep(
    sweep_params,
    sweep_description='This is a test run.'
)
