import pathlib as path

from .lib import data
from .lib.utils import misc
from .lib.utils import graph

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
    sweep_name,
    run_names,
    figure_prefix_list,
    ms_per_frame=100,
    experiment_folder=data.EXPERIMENT_FOLDER
):

    for figure_prefix in figure_prefix_list:

        animate_from_filenames(
            ['-'.join([figure_prefix, run_name]) for run_name in run_names],
            '-'.join(['animation', figure_prefix, sweep_name]),
            ms_per_frame=ms_per_frame,
            input_folder_or_list=[path.Path()/experiment_folder/sweep_name/run_name for run_name in run_names],
            output_folder=path.Path()/experiment_folder/sweep_name
        )
        
def generate_sweep_animations(
    sweep_name,
    animation_param,
    is_multiagent=False,
    figure_prefix=None,
    fixed_param_values={},
    ms_per_frame=100,
    experiment_folder=data.EXPERIMENT_FOLDER
):
    sweep = data.load_full_sweep(sweep_name, folder=experiment_folder)
    mdp_param_key = 'mdp_graph_agent_A' if is_multiagent else 'mdp_graph'

    run_names = [
        misc.build_run_name(
            sweep_name,
            { animation_param: anim_value, **fixed_param_values },
            list(fixed_param_values.keys()) + [animation_param]
        ) for anim_value in sweep['parameters'][animation_param]['values']
    ]

    try:

        mdp_key = sweep['parameters'][mdp_param_key]['value']
    except KeyError:
        mdp_key = sweep['parameters'][mdp_param_key]['values'][0][0]

    figure_prefix_list = [figure_prefix] if figure_prefix is not None else [
        'POWER_means',
        'POWER_samples'
    ] + [
        'POWER_correlations_{}'.format(state) for state in graph.get_states_from_graph(
            data.load_graph_from_dot_file(mdp_key)
        )
    ]

    animate_full_sweep(
        sweep_name,
        run_names,
        figure_prefix_list,
        ms_per_frame=ms_per_frame,
        experiment_folder=experiment_folder
    )


