import wandb as wb
import os
import json
import pathlib as path

from src.lib.utils import dist
from src.lib.utils import misc
from src.lib import data
from src.lib import get
from src.lib import save
from src.lib import runex
from src import viz

def cli_experiment_sweep(
    distribution_dict=dist.DISTRIBUTION_DICT,
    local_sweep_name=os.environ.get('LOCAL_SWEEP_NAME'),
    sweep_variable_params=json.loads(os.environ.get('SWEEP_VARIABLE_PARAMS')),
    plot_as_gridworld=(os.environ.get('PLOT_AS_GRIDWORLD') == 'True'),
    plot_distributions=(os.environ.get('PLOT_DISTRIBUTIONS') == 'True'),
    plot_correlations=(os.environ.get('PLOT_CORRELATIONS') == 'True'),
    diagnostic_mode=(os.environ.get('DIAGNOSTIC_MODE') == 'True'),
    plot_in_log_scale=(os.environ.get('PLOT_IN_LOG_SCALE') == 'True'),
    experiment_folder=data.EXPERIMENT_FOLDER,
    mdps_folder=data.MDPS_FOLDER,
    policies_folder=data.POLICIES_FOLDER,
    graph_descriptions=save.ALL_GRAPH_DESCRIPTIONS
):

    with wb.init() as run:
        run_params = {
            key: (value[0] if (key in sweep_variable_params) else value) for key, value in run.config.items()
        }
        # Hack to force a specified name for the run in W&B interface
        run.name = misc.build_run_name(local_sweep_name, run.config, sweep_variable_params)
        sweep_type = misc.determine_sweep_type(run_params)

        save_folder = path.Path(experiment_folder)/local_sweep_name/run.name
        data.create_folder(save_folder)

        num_reward_samples = run_params.get('num_reward_samples')
        num_workers = run_params.get('num_workers')
        discount_rate_agent_A = run_params.get('discount_rate')
        discount_rate_agent_B = run_params.get('discount_rate_agent_B')
        convergence_threshold = run_params.get('convergence_threshold')
        random_seed = run_params.get('random_seed')
        reward_correlation = run_params.get('reward_correlation')

        transition_graphs = get.get_transition_graphs(
            run_params, sweep_type, mdps_folder=mdps_folder, policies_folder=policies_folder
        )

        state_list, graphs_to_plot = save.save_graphs_and_generate_data_from_sweep_type(
            transition_graphs,
            run.name,
            sweep_type,
            save_folder=save_folder,
            graph_descriptions=graph_descriptions
        )

        reward_sampler = dist.config_to_reward_distribution(
            state_list, run_params.get('reward_distribution'), distribution_dict=distribution_dict
        )

        symmetric_interval = dist.config_to_symmetric_interval(
            run_params.get('reward_distribution')['default_dist'], distribution_dict=distribution_dict
        )

        (
            reward_samples_agent_A,
            reward_samples_agent_B,
            power_samples_agent_A,
            power_samples_agent_B,
            diagnostic_dict
        ) = runex.run_one_experiment(
            transition_graphs,
            discount_rate_agent_A,
            reward_sampler,
            sweep_type=sweep_type,
            discount_rate_agent_B=discount_rate_agent_B,
            num_reward_samples=num_reward_samples,
            num_workers=num_workers,
            convergence_threshold=convergence_threshold,
            random_seed=random_seed,
            reward_correlation=reward_correlation,
            symmetric_interval=symmetric_interval,
            diagnostic_mode=diagnostic_mode
        )

        data.save_experiment({
            'name': run.name,
            'inputs': { 'num_reward_samples_actual': len(reward_samples_agent_A), **run_params },
            'outputs': {
                'reward_samples': reward_samples_agent_A,
                'reward_samples_agent_B': reward_samples_agent_B,
                'power_samples': power_samples_agent_A,
                'power_samples_agent_B': power_samples_agent_B
            },
            'diagnostics': diagnostic_dict
        }, folder=save_folder)

        viz_kwargs = {
            'save_figure': data.save_figure,
            'save_folder': save_folder,
            'show': False,
            'plot_as_gridworld': plot_as_gridworld,
            'plot_distributions': plot_distributions,
            'plot_correlations': plot_correlations,
            'plot_in_log_scale': plot_in_log_scale
        }

        run.log({
            fig_name: wb.Image(fig) for fig_name, fig in viz.plot_all_outputs(
                reward_samples_agent_A,
                power_samples_agent_A,
                graphs_to_plot,
                save_handle='agent_A-{}'.format(run.name),
                **viz_kwargs
            ).items()
        })

        if sweep_type == 'multiagent_with_reward':
            run.log({
                fig_name: wb.Image(fig) for fig_name, fig in viz.plot_all_outputs(
                    reward_samples_agent_B,
                    power_samples_agent_B,
                    [graphs_to_plot[0]],
                    save_handle='agent_B-{}'.format(run.name),
                    **viz_kwargs
                ).items()
            })

            if diagnostic_mode:
                run.log({
                    fig_name: wb.Image(fig) for fig_name, fig in viz.plot_all_outputs(
                        reward_samples_agent_A,
                        diagnostic_dict['power_samples_A_against_seed'],
                        graphs_to_plot,
                        save_handle='agent_A_AGAINST_SEED-{}'.format(run.name),
                        **viz_kwargs
                    ).items()
                })

                run.log({
                    fig_name: wb.Image(fig) for fig_name, fig in viz.plot_all_outputs(
                        reward_samples_agent_B,
                        diagnostic_dict['power_samples_B_random'],
                        [graphs_to_plot[0]],
                        save_handle='agent_B_SEED_POLICY-{}'.format(run.name),
                        **viz_kwargs
                    ).items()
                })

if __name__ == '__main__':
    cli_experiment_sweep()