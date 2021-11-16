import pathlib as path
import dill
import torch
import json
import torch.distributions as td

import utils

################################################################################

EXPERIMENT_FOLDER = 'expts'
SETTINGS_FILENAME = 'settings.json'

WANDB_KEY_PATH = 'private.WANDB_API_KEY'
WANDB_ENTITY_PATH = 'public.WANDB_DEFAULT_ENTITY'

################################################################################

STATE_LIST = ['★', '∅', 'ℓ_◁', 'ℓ_↖', 'ℓ_↙', 'r_▷', 'r_↗', 'r_↘', 'TERMINAL']

DISTRIBUTION_DICT = {
    'uniform': td.Uniform,
    'uniform_0_1_manual': utils.pdf_sampler_constructor(pdf=lambda x: 1, interval=(0, 1), resolution=100)
}

ADJACENCY_MATRIX_DICT = {
    'mdp_from_paper': torch.tensor([
    #    ★  ∅ ℓ◁ ℓ↖ ℓ↙ r▷ r↗ r↘  T
        [0, 1, 1, 0, 0, 1, 0, 0, 0], # ★
        [0, 1, 0, 0, 0, 0, 0, 0, 0], # ∅
        [0, 0, 0, 1, 1, 0, 0, 0, 0], # ℓ_◁
        [0, 0, 0, 0, 1, 0, 0, 0, 1], # ℓ_↖
        [0, 0, 0, 1, 1, 0, 0, 0, 0], # ℓ_↙
        [0, 0, 0, 0, 0, 0, 1, 1, 0], # r_▷
        [0, 0, 0, 0, 0, 0, 1, 1, 0], # r_↗
        [0, 0, 0, 0, 0, 0, 1, 1, 0], # r_↘
        [0, 0, 0, 0, 0, 0, 0, 0, 1]  # TERMINAL
    ])
}

################################################################################

def save_experiment(experiment, folder=EXPERIMENT_FOLDER):
    path.Path.mkdir(path.Path()/folder, exist_ok=True)

    with open(
        path.Path()/folder/'data-{}.dill'.format(experiment['name']), 'wb'
    ) as f:
        dill.dump(experiment, f)

def load_experiment(experiment_name, folder=EXPERIMENT_FOLDER):
    with open(path.Path()/folder/experiment_name/'data-{}.dill'.format(experiment_name), 'rb') as f:
        experiment = dill.load(f)
    
    return experiment

def save_figure(figure, fig_name, folder=EXPERIMENT_FOLDER):
    path.Path.mkdir(path.Path()/folder, exist_ok=True)
    figure.savefig(path.Path()/folder/'{}.png'.format(fig_name), transparent=True)

def get_settings_value(settings_key_path, settings_filename=SETTINGS_FILENAME):
    with open(path.Path()/settings_filename, 'rb') as f:
        value = utils.retrieve(json.load(f), settings_key_path)
    
    return value