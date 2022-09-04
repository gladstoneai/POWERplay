import pathlib as path

from .lib import data
from .lib import get
from .lib.utils import misc

def animate_from_filenames(
    list_of_filenames,
    output_gif_filename,
    ms_per_frame=100,
    input_folder_or_list=data.TEMP_FOLDER,
    output_folder=data.TEMP_FOLDER
):
    input_folder_list = input_folder_or_list if (
        isinstance(input_folder_or_list, list)
    ) else [input_folder_or_list] * len(list_of_filenames)

    data.save_gif_from_frames(
        [
            data.load_png_figure(
                filename, input_folder
            ) for filename, input_folder in zip(list_of_filenames, input_folder_list)
        ],
        output_gif_filename,
        output_folder,
        ms_per_frame=ms_per_frame
    )

def animate_rollout(
    rollout_handle,
    number_of_steps,
    output_filename='rollout_animation',
    ms_per_frame=100,
    input_folder=data.TEMP_FOLDER,
    output_folder=data.TEMP_FOLDER
):
    animate_from_filenames(
        ['{0}-{1}'.format(rollout_handle, i) for i in range(number_of_steps)],
        output_filename,
        ms_per_frame=ms_per_frame,
        input_folder_or_list=input_folder,
        output_folder=output_folder
    )

def animate_full_sweep(
    sweep_id,
    run_suffixes,
    figure_prefix_list,
    ms_per_frame=100,
    experiment_folder=data.EXPERIMENT_FOLDER
):
    sweep_name = data.get_full_sweep_name_from_id(sweep_id)
    run_names = [misc.build_run_name_from_run_suffix(sweep_name, run_suffix) for run_suffix in run_suffixes]

    for figure_prefix in figure_prefix_list:
        
        try:
            animate_from_filenames(
                ['-'.join([figure_prefix, run_name]) for run_name in run_names],
                '-'.join(['animation', figure_prefix, sweep_name]),
                ms_per_frame=ms_per_frame,
                input_folder_or_list=[path.Path()/experiment_folder/sweep_name/run_name for run_name in run_names],
                output_folder=path.Path()/experiment_folder/sweep_name
            )
        
        except FileNotFoundError:
            print(
                'No image files corresponding to figure {0} of sweep {1}; skipping animation.'.format(
                    figure_prefix, sweep_id
                )
            )
        
def generate_sweep_animations(
    sweep_id,
    animation_param,
    figure_prefix=None,
    ms_per_frame=100,
    experiment_folder=data.EXPERIMENT_FOLDER
):
    if figure_prefix is None:
        state_list = get.get_sweep_state_list(sweep_id, folder=experiment_folder)
        sweep_type = get.get_sweep_type(sweep_id, folder=experiment_folder)

        if sweep_type == 'single_agent' or sweep_type == 'multiagent_fixed_policy':
            figure_prefix_list_ = [
                'POWER_means', 'POWER_samples'
            ] + [
                'POWER_correlations_{}'.format(state) for state in state_list
            ]
        
        elif sweep_type == 'multiagent_with_reward':
            figure_prefix_list_ = [
                'POWER_means-agent_A', 'POWER_means-agent_B', 'POWER_samples-agent_A', 'POWER_samples-agent_B'
            ] + [
                'POWER_correlations_{}-agent_A'.format(state) for state in state_list
            ] + [
                'POWER_correlations_{}-agent_B'.format(state) for state in state_list
            ]

    else:
        figure_prefix_list_ = [figure_prefix]

    animate_full_sweep(
        sweep_id,
        get.get_sweep_run_suffixes_for_param(sweep_id, animation_param, folder=experiment_folder),
        figure_prefix_list_,
        ms_per_frame=ms_per_frame,
        experiment_folder=experiment_folder
    )


