import pathlib as path
import time
import subprocess as sp
import re
import os
import json

from .lib.utils import misc
from .lib import data
from .lib import check

def launch_sweep(
    sweep_config_filename,
    sweep_local_id=time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())),
    entity=data.get_settings_value(data.WANDB_ENTITY_PATH, settings_filename=data.SETTINGS_FILENAME),
    project='uncategorized',
    sweep_config_folder=data.SWEEP_CONFIGS_FOLDER,
    output_folder_local=data.EXPERIMENT_FOLDER,
    plot_as_gridworld=False,
    plot_correlations=True,
    beep_when_done=False,
    environ=os.environ
):
    input_sweep_config = data.load_sweep_config(sweep_config_filename, folder=sweep_config_folder)
    check.check_sweep_params(input_sweep_config.get('parameters'))

    sweep_name = '{0}-{1}'.format(sweep_local_id, input_sweep_config.get('name'))

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
        'PLOT_CORRELATIONS': str(plot_correlations)
    })

    if beep_when_done:
        print('\a')
        print('\a')
        print('\a')
