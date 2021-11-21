import pathlib as path
import dill
import torch
import json
import torch.distributions as td
import yaml

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

def save_experiment(experiment, folder=EXPERIMENT_FOLDER): # TODO: Change name of this function to be more general
    path.Path.mkdir(path.Path()/folder, exist_ok=True)

    with open(path.Path()/folder/'data-{}.dill'.format(experiment['name']), 'wb') as f:
        dill.dump(experiment, f)

def load_experiment(experiment_name, folder=EXPERIMENT_FOLDER): # TODO: Delete
    with open(path.Path()/folder/experiment_name/'data-{}.dill'.format(experiment_name), 'rb') as f:
        experiment = dill.load(f)
    
    return experiment

def load_sweep_config(sweep_config_filename, folder=EXPERIMENT_FOLDER):
    with open(path.Path()/folder/sweep_config_filename, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    return sweep_config

def save_sweep_config(sweep_config_dict, folder=EXPERIMENT_FOLDER):
    path.Path.mkdir(path.Path()/folder, exist_ok=True)
    sweep_config_filepath = path.Path()/folder/'{}.yaml'.format(sweep_config_dict.get('name'))

    with open(sweep_config_filepath, 'w') as f:
        yaml.dump(sweep_config_dict, f, allow_unicode=True)
    
    return sweep_config_filepath

def load_full_sweep(sweep_name, folder=EXPERIMENT_FOLDER):
    sweep_dict = load_sweep_config('{}.yaml'.format(sweep_name), folder=path.Path(folder)/sweep_name)
    runs_dict = {}
    for item_path in (path.Path()/folder/sweep_name).iterdir():
        if item_path.is_dir():
            with open(item_path/'data-{}.dill'.format(item_path.name), 'rb') as fr:
                runs_dict[item_path.name] = dill.load(fr)

    return {
        **sweep_dict,
        'all_runs_data': runs_dict
    }

def save_figure(figure, fig_name, folder=EXPERIMENT_FOLDER):
    path.Path.mkdir(path.Path()/folder, exist_ok=True)
    figure.savefig(path.Path()/folder/'{}.png'.format(fig_name), transparent=True)

def get_settings_value(settings_key_path, settings_filename=SETTINGS_FILENAME):
    with open(path.Path()/settings_filename, 'rb') as f:
        value = utils.retrieve(json.load(f), settings_key_path)
    
    return value
