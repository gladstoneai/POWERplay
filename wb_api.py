import wandb as wb

import data

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
