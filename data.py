import pathlib as path
import dill
import torch
import functools as func
import operator as op
import json

################################################################################

EXPERIMENT_FOLDER = 'expts'
SETTINGS_FILENAME = 'settings.json'
WANDB_KEY_PATH = 'private.WANDB_API_KEY'

################################################################################

STATE_LIST = ['★', '∅', 'ℓ_◁', 'ℓ_↖', 'ℓ_↙', 'r_▷', 'r_↗', 'r_↘', 'TERMINAL']
ADJACENCY_MATRIX = torch.tensor([
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

################################################################################

def save_experiment(experiment, folder=EXPERIMENT_FOLDER):
    path.Path.mkdir(path.Path()/folder, exist_ok=True)

    with open(
        path.Path()/folder/'data-{}.dill'.format(experiment['name']), 'wb'
    ) as f:
        dill.dump(experiment, f)

def load_experiment(file_name, folder=EXPERIMENT_FOLDER):
    with open(path.Path()/folder/file_name, 'rb') as f:
        experiment = dill.load(f)
    
    return experiment

def save_figure(figure, fig_name, folder=EXPERIMENT_FOLDER):
    path.Path.mkdir(path.Path()/folder, exist_ok=True)
    figure.savefig(path.Path()/folder/'{}.png'.format(fig_name), transparent=True)

def get_settings_value(settings_key_path, settings_filename=SETTINGS_FILENAME):
    with open(path.Path()/settings_filename, 'rb') as f:
        value = func.reduce(op.getitem, settings_key_path.split('.'), json.load(f))
    
    return value