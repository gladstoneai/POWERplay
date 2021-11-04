import pathlib as path
import dill
import torch

################################################################################

EXPERIMENT_FOLDER = 'expts'

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

    with open(path.Path()/folder/'{}.dill'.format(experiment['name']), 'wb') as f:
        dill.dump(experiment, f)

def load_experiment(file_name, folder=EXPERIMENT_FOLDER):
    with open(path.Path()/folder/file_name, 'rb') as f:
        experiment = dill.load(f)
    
    return experiment

def save_figure(figure, fig_name, folder=EXPERIMENT_FOLDER):
    figure.savefig(path.Path()/folder/'{}.png'.format(fig_name), transparent=True)