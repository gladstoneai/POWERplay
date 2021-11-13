import wandb as wb

import data
import viz

def login():
    wb.login(key=data.get_settings_value(data.WANDB_KEY_PATH))

def initialize_run(experiment_name, wandb_run_params, experiment_config):
    return wb.init(**{
        **wandb_run_params,
        **{
            'name': experiment_name,
            'config': experiment_config
        }
    })

def push_to_wandb(experiment, project=None, entity=None, notes=None):
    login()

    with initialize_run(
        experiment['name'],
        { 'project': project, 'entity': entity, 'notes': notes },
        experiment['inputs']
    ) as tracker:

        viz.render_all_outputs(
            experiment['outputs']['reward_samples'],
            experiment['outputs']['power_samples'],
            experiment['inputs']['state_list'],
            show=False,
            save_fig=False,
            save_handle=None,
            save_folder=None,
            wb_tracker=tracker
        )
