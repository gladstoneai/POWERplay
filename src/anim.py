import pathlib as path

from . import data
from . import utils

def animate_full_sweep(
    sweep_name,
    run_names,
    figure_prefix_list,
    experiment_folder=data.EXPERIMENT_FOLDER
):

    for figure_prefix in figure_prefix_list:
        
        png_frames = [
            data.load_png_figure(
                '-'.join([figure_prefix, run_name]), path.Path()/experiment_folder/sweep_name/run_name
            ) for run_name in run_names
        ]

        data.save_gif_from_frames(
            png_frames,
            '-'.join(['animation', figure_prefix, sweep_name]),
            path.Path()/experiment_folder/sweep_name
        )

def generate_sweep_animations(
    sweep_name,
    animation_param,
    figure_prefix=None,
    fixed_param_values={},
    experiment_folder=data.EXPERIMENT_FOLDER
):
    sweep = data.load_full_sweep(sweep_name, folder=experiment_folder)

    run_names = [
        utils.build_run_name(
            sweep_name,
            { animation_param: anim_value, **fixed_param_values },
            list(fixed_param_values.keys()) + [animation_param]
        ) for anim_value in sweep['parameters'][animation_param]['values']
    ]

    figure_prefix_list = [figure_prefix] if figure_prefix is not None else [
        'POWER_means',
        'POWER_samples'
    ] + [
        'POWER_correlations_{}'.format(state) for state in utils.get_states_from_graph(
            data.load_graph_from_dot_file(sweep['parameters']['mdp_graph']['value'])
        )[:-1]
    ]

    animate_full_sweep(sweep_name, run_names, figure_prefix_list, experiment_folder=experiment_folder)


