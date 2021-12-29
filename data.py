import pathlib as path
import dill
import json
import torch.distributions as td
import yaml
import networkx as nx
import PIL as pil
import graphviz as gv

import utils

################################################################################

SWEEP_CONFIGS_FOLDER = 'configs'
EXPERIMENT_FOLDER = 'expts'
MDPS_FOLDER = 'mdps'
TEMP_FOLDER = 'temp'

SETTINGS_FILENAME = 'settings.json'

WANDB_KEY_PATH = 'private.WANDB_API_KEY'
WANDB_ENTITY_PATH = 'public.WANDB_DEFAULT_ENTITY'

################################################################################

DISTRIBUTION_DICT = {
    'uniform': td.Uniform,
    'uniform_0_1_manual': utils.pdf_sampler_constructor(pdf=lambda x: 1, interval=(0, 1), resolution=100),
    'beta': td.Beta
}

################################################################################

def save_experiment(experiment, folder=EXPERIMENT_FOLDER):
    path.Path.mkdir(path.Path()/folder, exist_ok=True)

    with open(path.Path()/folder/'data-{}.dill'.format(experiment['name']), 'wb') as f:
        dill.dump(experiment, f)

def load_experiment(experiment_name, folder=EXPERIMENT_FOLDER):
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
    runs_dict_ = {}
    for item_path in (path.Path()/folder/sweep_name).iterdir():
        if item_path.is_dir():
            with open(item_path/'data-{}.dill'.format(item_path.name), 'rb') as fr:
                runs_dict_[item_path.name] = dill.load(fr)

    return {
        **sweep_dict,
        'all_runs_data': runs_dict_
    }

def save_figure(figure, fig_name, folder=EXPERIMENT_FOLDER):
    path.Path.mkdir(path.Path()/folder, exist_ok=True)
    figure.savefig(path.Path()/folder/'{}.png'.format(fig_name), transparent=True)

def get_settings_value(settings_key_path, settings_filename=SETTINGS_FILENAME):
    with open(path.Path()/settings_filename, 'rb') as f:
        value = utils.retrieve(json.load(f), settings_key_path)
    
    return value

def load_png_figure(figure_name, folder):
    return pil.Image.open(path.Path()/folder/'{}.png'.format(figure_name))

def save_gif_from_frames(frames_list, gif_name, folder):
    frames_list[0].save(
        path.Path()/folder/'{}.gif'.format(gif_name),
        format='GIF',
        append_images=frames_list,
        save_all=True,
        duration=100,
        loop=0
    )

def load_graph_from_dot_file(mdp_name, folder=MDPS_FOLDER):
    mdp_graph_ = nx.drawing.nx_pydot.read_dot(path.Path()/folder/'{}.gv'.format(mdp_name))

# NOTE: Either the dot format or the NetworkX pydot converter has a bug that often causes
# an extra orphan node to be appended to the extracted graph, whose name is '\\n'. This seems
# to be related to bad escaping of the '\' character in one of the steps. We remove those
# nodes manually in the extraction process when they occur.
    try:
        mdp_graph_.remove_node('\\n')
    except nx.exception.NetworkXError:
        pass

    return mdp_graph_

def save_graph_to_dot_file(mdp_graph, mdp_filename, folder=MDPS_FOLDER):
    nx.drawing.nx_pydot.write_dot(mdp_graph, path.Path()/folder/'{}.gv'.format(mdp_filename))

def create_and_save_mdp_figure(
    mdp_filename,
    fig_name=None,
    mdps_folder=MDPS_FOLDER,
    fig_folder=TEMP_FOLDER,
    show=False
):
    fig_filename = fig_name if fig_name is not None else mdp_filename
    fig_filepath = path.Path()/fig_folder/'{}.png'.format(fig_filename)

    gv.render(
        'dot',
        'png',
        path.Path()/mdps_folder/'{}.gv'.format(mdp_filename),
        outfile=fig_filepath
    )
    
    if show:
        gv.view(fig_filepath)

    return load_png_figure(fig_filepath.stem, fig_filepath.parent)