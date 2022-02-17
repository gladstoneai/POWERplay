import pathlib as path
import time

from . import launch
from . import data
from . import utils
from . import viz
from . import mdp

def augment_runs_data(sweep_runs_data, convert_mdp_to_stochastic=False, plot_as_gridworld=False):
    augmented_runs_data_ = {}

    for run_name, run_data in sweep_runs_data.items():
        raw_graph = data.load_graph_from_dot_file(run_data['inputs']['mdp_graph'])

        if convert_mdp_to_stochastic and plot_as_gridworld:
            mdp_graph = mdp.gridworld_to_stochastic_graph(raw_graph)
        elif convert_mdp_to_stochastic:
            mdp_graph = mdp.mdp_to_stochastic_graph(raw_graph)
        else:
            mdp_graph = raw_graph

        augmented_runs_data_[run_name] = {
            **run_data,
            'inputs': {
                **run_data['inputs'],
                'mdp_graph_networkx': mdp_graph,
                'transition_tensor': utils.graph_to_transition_tensor(mdp_graph)
            }
        }
    
    return augmented_runs_data_

def clone_run_inputs(sweep_runs_data, convert_mdp_to_stochastic=False, ignore_terminal_state=False):
    return utils.clone_run_inputs(
        augment_runs_data(sweep_runs_data, convert_mdp_to_stochastic=convert_mdp_to_stochastic),
        ignore_terminal_state=ignore_terminal_state
    )

def run_and_save_sweep_replication(
    sweep_name,
    ignore_terminal_state=False,
    convert_mdp_to_stochastic=False,
    experiment_folder=data.EXPERIMENT_FOLDER,
    test_local_id=time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())),
    plot_correlations=True,
    plot_as_gridworld=False
):
    replication_prefix = '{}-replication-'.format(test_local_id)
    replication_name = replication_prefix + sweep_name
    replication_folder = path.Path(experiment_folder)/replication_name
    data.create_folder(replication_folder)

    augmented_runs_data = augment_runs_data(
        data.load_full_sweep(sweep_name).get('all_runs_data'),
        convert_mdp_to_stochastic=convert_mdp_to_stochastic,
        plot_as_gridworld=plot_as_gridworld
    )
    all_run_inputs = utils.clone_run_inputs(
        augmented_runs_data, ignore_terminal_state=ignore_terminal_state
    )

    print('Running sweeps for replication:')
    print()
    print(replication_name)

    for run_name in all_run_inputs.keys():
        run_inputs = all_run_inputs[run_name]
        augmented_run_data = augmented_runs_data[run_name]
        save_folder = replication_folder/(replication_prefix + run_name)

        print()
        print()
        print(run_name)
        print()

        power_samples = launch.rewards_to_powers(*run_inputs['args'], **run_inputs['kwargs'])

        data.save_experiment({
            'name': replication_prefix + run_name,
            'inputs': {
                'reward_samples': run_inputs['args'][0],
                'mdp_graph': augmented_run_data['inputs']['mdp_graph'],
                'discount_rate': run_inputs['args'][2],
                **run_inputs['kwargs']
            },
            'outputs': {
                'power_samples': power_samples
            }
        }, folder=save_folder)

        viz.render_all_outputs(
            run_inputs['args'][0],
            power_samples,
            augmented_run_data['inputs']['mdp_graph_networkx'],
            plot_correlations=plot_correlations,
            save_handle=run_name,
            save_figure=data.save_figure,
            save_folder=save_folder,
            show=False,
            plot_as_gridworld=plot_as_gridworld
        )

def test_vanilla():
    launch.launch_sweep(
        'test_sweep.yaml',
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        project='uncategorized'
    )

def test_gridworld():
    launch.launch_sweep(
        'test_sweep_gridworld.yaml',
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        project='uncategorized',
        plot_as_gridworld=True
    )

def test_stochastic():
    launch.launch_sweep(
        'test_sweep_stochastic.yaml',
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        project='uncategorized'
    )
