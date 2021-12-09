import pathlib as path

import data
import utils

def generate_sweep_animations(
    sweep_name,
    figure_prefix,
    animation_param,
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

    png_frames = [
        data.load_png_figure(
            '-'.join([figure_prefix, run_name]), path.Path(experiment_folder)/sweep_name/run_name
        ) for run_name in run_names
    ]

    data.save_gif_from_frames(
        png_frames,
        '-'.join(['animation', figure_prefix, sweep_name]),
        path.Path(experiment_folder)/sweep_name
    )






    # TODO: Select the sweep variable param to animate over and construct the names of each png graph.




