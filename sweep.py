import wandb as wb
import os
import json
import pathlib as path

from src.lib.utils import dist
from src.lib.utils import misc
from src.lib.utils import graph
from src.lib import data
from src.lib import calc
from src import viz

def cli_experiment_sweep(
    distribution_dict=dist.DISTRIBUTION_DICT,
    local_sweep_name=os.environ.get('LOCAL_SWEEP_NAME'),
    sweep_variable_params=json.loads(os.environ.get('SWEEP_VARIABLE_PARAMS')),
    plot_as_gridworld=(os.environ.get('PLOT_AS_GRIDWORLD') == 'True'),
    plot_correlations=(os.environ.get('PLOT_CORRELATIONS') == 'True'),
    experiment_folder=data.EXPERIMENT_FOLDER,
    mdps_folder=data.MDPS_FOLDER,
    policies_folder=data.POLICIES_FOLDER
):

    with wb.init() as run:
        run_params = {
            key: (value[0] if (key in sweep_variable_params) else value) for key, value in run.config.items()
        }
        # Hack to force a specified name for the run in W&B interface
        run.name = misc.build_run_name(local_sweep_name, run.config, sweep_variable_params)

        save_folder = path.Path(experiment_folder)/local_sweep_name/run.name
        data.create_folder(save_folder)

        num_reward_samples = run_params.get('num_reward_samples')
        num_workers = run_params.get('num_workers')
        discount_rate = run_params.get('discount_rate')
        convergence_threshold = run_params.get('convergence_threshold')
        random_seed = run_params.get('random_seed')
        transition_tensor, graphs_output = calc.compute_transition_tensor(
            run_params, mdps_folder=mdps_folder, policies_folder=policies_folder
        )

        if len(graphs_output) == 1: # Single agent case
            mdp_graph = graphs_output[0]

            data.save_graph_to_dot_file(mdp_graph, 'mdp_graph-{}'.format(run.name), folder=save_folder)
            state_list = graph.get_states_from_graph(mdp_graph)
            graphs_to_plot = [
                { 'graph_name': 'MDP graph', 'graph_type': 'mdp_graph', 'graph_data': mdp_graph }
            ]

        else: # Multiagent case
            mdp_graph_A, mdp_graph_B, policy_graph_B = graphs_output

            data.save_graph_to_dot_file(
                mdp_graph_A, 'mdp_graph_agent_A-{}'.format(run.name), folder=save_folder
            )
            data.save_graph_to_dot_file(
                mdp_graph_B, 'mdp_graph_agent_B-{}'.format(run.name), folder=save_folder
            )
            data.save_graph_to_dot_file(
                policy_graph_B, 'policy_graph_agent_B-{}'.format(run.name), folder=save_folder
            )
            state_list = graph.get_states_from_graph(mdp_graph_A)
            graphs_to_plot = [
                {
                    'graph_name': 'MDP graph for agent A',
                    'graph_type': 'mdp_graph_agent_A',
                    'graph_data': mdp_graph_A
                }, {
                    'graph_name': 'MDP graph for agent B',
                    'graph_type': 'mdp_graph_agent_B',
                    'graph_data': mdp_graph_B
                }, {
                    'graph_name': 'Policy graph for agent B',
                    'graph_type': 'policy_graph_agent_B',
                    'graph_data': policy_graph_B
                }
            ]

        reward_sampler = dist.config_to_reward_distribution(
            state_list, run_params.get('reward_distribution'), distribution_dict=distribution_dict
        )

        reward_samples, power_samples = calc.run_one_experiment(
            transition_tensor,
            discount_rate,
            reward_sampler,
            num_reward_samples=num_reward_samples,
            num_workers=num_workers,
            convergence_threshold=convergence_threshold,
            random_seed=random_seed
        )

        data.save_experiment({
            'name': run.name,
            'inputs': { 'num_reward_samples_actual': len(reward_samples), **run_params },
            'outputs': {
                'reward_samples': reward_samples, 'power_samples': power_samples
            }
        }, folder=save_folder)

        run.log({ fig_name: wb.Image(fig) for fig_name, fig in viz.plot_all_outputs(
            reward_samples,
            power_samples,
            graphs_to_plot,
            save_handle=run.name,
            save_figure=data.save_figure,
            save_folder=save_folder,
            show=False,
            plot_as_gridworld=plot_as_gridworld,
            plot_correlations=plot_correlations
        ).items() })

if __name__ == '__main__':
    cli_experiment_sweep()