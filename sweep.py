import wandb as wb
import os
import json
import pathlib as path

from src import data
from src import viz
from src import dist
from src import launch
from src import utils

def cli_experiment_sweep(
    distribution_dict=data.DISTRIBUTION_DICT,
    local_sweep_name=os.environ.get('LOCAL_SWEEP_NAME'),
    sweep_variable_params=json.loads(os.environ.get('SWEEP_VARIABLE_PARAMS')),
    plot_as_gridworld=(os.environ.get('PLOT_AS_GRIDWORLD') == 'True'),
    plot_correlations=(os.environ.get('PLOT_CORRELATIONS') == 'True'),
    experiment_folder=data.EXPERIMENT_FOLDER,
    mdps_folder=data.MDPS_FOLDER
):

    with wb.init() as run:
        run_params = {
            key: (value[0] if (key in sweep_variable_params) else value) for key, value in run.config.items()
        }
        # Hack to force a specified name for the run in W&B interface
        run.name = utils.build_run_name(local_sweep_name, run.config, sweep_variable_params)

        num_reward_samples = run_params.get('num_reward_samples')
        num_workers = run_params.get('num_workers')
        discount_rate = run_params.get('discount_rate')
        convergence_threshold = run_params.get('convergence_threshold')
        random_seed = run_params.get('random_seed')

        mdp_graph = data.load_graph_from_dot_file(run_params.get('mdp_graph'), folder=mdps_folder)
        state_action_matrix = utils.graph_to_state_action_matrix(mdp_graph)
        reward_sampler = dist.config_to_reward_distribution(
            list(mdp_graph), run_params.get('reward_distribution'), distribution_dict=distribution_dict
        )

        reward_samples, power_samples = launch.run_one_experiment(
            state_action_matrix,
            discount_rate,
            reward_sampler,
            num_reward_samples=num_reward_samples,
            num_workers=num_workers,
            convergence_threshold=convergence_threshold,
            random_seed=random_seed
        )

        save_folder = path.Path(experiment_folder)/local_sweep_name/run.name

        data.save_experiment({
            'name': run.name,
            'inputs': { 'num_reward_samples_actual': len(reward_samples), **run_params },
            'outputs': {
                'reward_samples': reward_samples, 'power_samples': power_samples
            }
        }, folder=save_folder)

        run.log({ fig_name: wb.Image(fig) for fig_name, fig in viz.render_all_outputs(
            reward_samples,
            power_samples,
            mdp_graph,
            save_handle=run.name,
            save_figure=data.save_figure,
            save_folder=save_folder,
            show=False,
            plot_as_gridworld=plot_as_gridworld,
            plot_correlations=plot_correlations
        ).items() })

if __name__ == '__main__':
    cli_experiment_sweep()