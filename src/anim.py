import pathlib as path

from .lib import data
from .lib.utils import misc
from .lib.utils import graph

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
    is_multiagent=False,
    figure_prefix=None,
    fixed_param_values={},
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

    animate_full_sweep(sweep_name, run_names, figure_prefix_list, experiment_folder=experiment_folder)


