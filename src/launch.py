import pathlib as path
import subprocess as sp
import re
import os
import json

from .lib.utils import misc
from .lib import data
from .lib import check

def launch_sweep(
    sweep_config_filename,
    sweep_local_id=misc.generate_sweep_id(),
    entity=data.get_settings_value(data.WANDB_ENTITY_PATH, settings_filename=data.SETTINGS_FILENAME),
    project='uncategorized',
    sweep_config_folder=data.SWEEP_CONFIGS_FOLDER,
    output_folder_local=data.EXPERIMENT_FOLDER,
    plot_as_gridworld=False,
    plot_distributions=False,
    plot_correlations=False,
    announce_when_done=False,
    diagnostic_mode=False,
    plot_in_log_scale=False,
    force_basic_font=False,
    environ=os.environ
):
    input_sweep_config = data.load_sweep_config(sweep_config_filename, folder=sweep_config_folder)
    check.check_sweep_params(input_sweep_config['parameters'])

    sweep_name = '{0}-{1}'.format(sweep_local_id, input_sweep_config['name'])

    output_sweep_config = misc.build_sweep_config(
        input_sweep_config=input_sweep_config,
        sweep_name=sweep_name,
        project=project,
        entity=entity
    )
    sweep_config_filepath = data.save_sweep_config(
        output_sweep_config, folder=path.Path(output_folder_local)/sweep_name
    )

# NOTE: The wandb Python API uses multithreading, which makes it incompatible with matplotlib.pyplot rendering.
# The wandb CLI does not use multithreading, and is compatible with pyplot, so we run the CLI via Python subprocess.
# (This behavior by wandb is undocumented.)
    sp.run(['wandb', 'login'])
    sp.run([
        'wandb',
        'agent',
        re.search(
            r'(?<=wandb agent )(.*)',
            sp.run(['wandb', 'sweep', sweep_config_filepath], capture_output=True).stderr.decode('utf-8')
        ).group()
    ], env={
        **environ,
        'LOCAL_SWEEP_NAME': sweep_name,
        'SWEEP_VARIABLE_PARAMS': json.dumps(misc.get_variable_params(input_sweep_config)),
        'PLOT_AS_GRIDWORLD': str(plot_as_gridworld), # Only str allowed in os.environ
        'PLOT_DISTRIBUTIONS': str(plot_distributions),
        'PLOT_CORRELATIONS': str(plot_correlations),
        'DIAGNOSTIC_MODE': str(diagnostic_mode),
        'PLOT_IN_LOG_SCALE': str(plot_in_log_scale),
        'FORCE_BASIC_FONT': str(force_basic_font)
    })

    if announce_when_done:
        sp.run(['say', '"Run complete"'])
