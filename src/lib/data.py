import pathlib as path
import dill
import json
import yaml
import networkx as nx
import PIL as pil
import graphviz as gv
import subprocess as sp
import os

from .utils import misc

################################################################################

SWEEP_CONFIGS_FOLDER = 'configs'
EXPERIMENT_FOLDER = 'expts'
MDPS_FOLDER = 'mdps'
POLICIES_FOLDER = 'policies'
TEMP_FOLDER = 'temp'

SETTINGS_FILENAME = 'settings.json'

WANDB_KEY_PATH = 'private.WANDB_API_KEY'
WANDB_ENTITY_PATH = 'public.WANDB_DEFAULT_ENTITY'

################################################################################

def create_folder(folder_name):
    path.Path.mkdir(path.Path()/folder_name, exist_ok=True)

def save_experiment(experiment, folder=EXPERIMENT_FOLDER):
    create_folder(folder)

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
    create_folder(folder)
    sweep_config_filepath = path.Path()/folder/'{}.yaml'.format(sweep_config_dict['name'])

    with open(sweep_config_filepath, 'w') as f:
        yaml.dump(sweep_config_dict, f, allow_unicode=True)
    
    return sweep_config_filepath

def load_full_sweep(sweep_name, folder=EXPERIMENT_FOLDER):

    try:
        sweep_dict = load_sweep_config(
            '{}.yaml'.format(sweep_name), folder=path.Path(folder)/sweep_name
        )
    except FileNotFoundError:
        sweep_dict = {}

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
    create_folder(folder)
    figure.savefig(path.Path()/folder/'{}.png'.format(fig_name), transparent=False)

def get_settings_value(settings_key_path, settings_filename=SETTINGS_FILENAME):
    with open(path.Path()/settings_filename, 'rb') as f:
        value = misc.retrieve(json.load(f), settings_key_path)
    
    return value

def load_png_figure(figure_name, folder):
    # We have to load the image and close the context manager (using the with block)
    # explicitly, because otherwise when saving multiple images to wandb the wb.Image()
    # function crashes due to lazy loading by PIL. See
    # https://pillow.readthedocs.io/en/stable/reference/open_files.html#file-handling
    with pil.Image.open(path.Path()/folder/'{}.png'.format(figure_name)) as img:
        img.load()
    
    return img

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
    input_graph_ = nx.drawing.nx_pydot.read_dot(path.Path()/folder/'{}.gv'.format(mdp_name))

# NOTE: Either the dot format or the NetworkX pydot converter has a bug that often causes
# an extra orphan node to be appended to the extracted graph, whose name is '\\n'. This seems
# to be related to bad escaping of the '\' character in one of the steps. We remove those
# nodes manually in the extraction process when they occur.
    try:
        input_graph_.remove_node('\\n')
    except nx.exception.NetworkXError:
        pass

# The following lines fix two issues with the dot format:
# 1) The dot format converts float weights to double-quoted strings in the saved file, which
# then get re-quoted when NetworkX loads from the dot file. This results in weights like
# '"0.9"'. So we have to strip away the " character and manually revert each weight to
# a float.
# 2) The dot format adds a trailing 0 to the tuple for each edge. Before saving, an edge
# would be ('1', '2'). After saving and re-loading, the same edge would be ('1', '2', 0).
# So we also have to create a whole new graph with the correct edge ids.
    output_graph_ = nx.DiGraph()
    output_graph_.add_nodes_from(input_graph_.nodes)
    output_graph_.add_edges_from([edge[:2] for edge in input_graph_.edges])

    nx.set_edge_attributes(
        output_graph_,
        {
            edge[:2]: float(weight.strip('"')) for edge, weight in nx.get_edge_attributes(
                input_graph_, 'weight'
            ).items()
        },
        name='weight'
    )
    
    return output_graph_

def save_graph_to_dot_file(mdp_graph, mdp_filename, folder=MDPS_FOLDER):
    nx.drawing.nx_pydot.write_dot(mdp_graph, path.Path()/folder/'{}.gv'.format(mdp_filename))

def create_and_save_mdp_figure(
    mdp_filename,
    subgraphs_per_row=4,
    fig_name=None,
    mdps_folder=MDPS_FOLDER,
    fig_folder=TEMP_FOLDER,
    show=False
):
    fig_filename = fig_name if fig_name is not None else mdp_filename
    fig_filepath = path.Path()/fig_folder/'{}.png'.format(fig_filename)

# See https://stackoverflow.com/questions/8002352/how-to-control-subgraphs-layout-in-dot
    proc1 = sp.Popen(['ccomps', '-x', path.Path()/mdps_folder/'{}.gv'.format(mdp_filename)], stdout=sp.PIPE)
    proc2 = sp.Popen(['dot'], stdin=proc1.stdout, stdout=sp.PIPE)
    proc3 = sp.Popen(['gvpack', '-array{}'.format(subgraphs_per_row)], stdin=proc2.stdout, stdout=sp.PIPE)
# Final command is a run to ensure execution is synchronous. This forces the process to wait until
# the fig file exists before trying to retreive it with gv.view or load_png_figure.
    sp.run(['neato', '-Tpng', '-n2', '-o', fig_filepath], stdin=proc3.stdout)
    
    if show:
        gv.view(fig_filepath)

    return load_png_figure(fig_filepath.stem, fig_filepath.parent)

def get_full_sweep_name_from_id(sweep_id, folder=EXPERIMENT_FOLDER):
    return [name for name in os.listdir(folder) if (
        name[:len(sweep_id)] == sweep_id
    )][0]
