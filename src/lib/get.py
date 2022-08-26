from .utils import graph
from .utils import misc
from . import data

def get_sweep_run_suffixes_for_param(sweep_id, param_name, folder=data.EXPERIMENT_FOLDER):
    sweep_name = data.get_full_sweep_name_from_id(sweep_id, folder=folder)
    sweep_params = data.load_full_sweep(sweep_name, folder=folder)['parameters']

    return [misc.build_run_suffix(
        { param_name: sweep_params[param_name]['values'][i] }, [param_name]
    ) for i in range(len(sweep_params[param_name]['values']))]

# TODO: Document this function. Lets you quickly retrieve inputs or outputs of an experiment based on
# the sweep_id (i.e., date & time hash, which should be unique) and the suffix for the run within
# the experiment sweep. Returns outputs by default. For inputs, set results_type='inputs'.
# For diagnostics, set results_type='diagnostics'.
def get_sweep_run_results(sweep_id, run_suffix, results_type='outputs', folder=data.EXPERIMENT_FOLDER):
    sweep_name = data.get_full_sweep_name_from_id(sweep_id, folder=folder)

    return data.load_full_sweep(
        sweep_name, folder=folder
    )['all_runs_data']['{0}-{1}'.format(sweep_name, run_suffix)].get(results_type, {})

# TODO: Document this function. This is a convenience function that lets you quickly retrieve
# the state_list for a sweep. IMPORTANT: This assumes that the state_list will be the same for all
# runs in the sweep, even though the MDP itself may change. This seems sensible, since if you're sweeping
# across runs you'll generally only be changing the transition probabilities of the MDP as opposed to the
# list of states itself, but I'm not 100% sure if this is true. Inputs are
# the sweep_id (i.e., date & time hash, which should be unique) and the suffix for the run within
# the experiment sweep.
def get_sweep_state_list(sweep_id, folder=data.EXPERIMENT_FOLDER):
    parameters = data.load_full_sweep(
        data.get_full_sweep_name_from_id(sweep_id, folder=folder), folder=folder
    ).get('parameters')
    mdp_graph_param = parameters.get('mdp_graph', parameters.get('mdp_graph_agent_A'))

    if mdp_graph_param.get('value'):
        state_list = graph.get_states_from_graph(
            data.load_graph_from_dot_file(mdp_graph_param['value'])
        )
    
    else:
        all_state_lists = [
            graph.get_states_from_graph(
                data.load_graph_from_dot_file(mdp_pair[0])
            ) for mdp_pair in mdp_graph_param['values']
        ]
        if all(all_state_lists[0] == item for item in all_state_lists):
            state_list = all_state_lists[0]

    return state_list

# TODO: Document.
# NOTE: The output of get_sweep_run_results(sweep_id, run_suffix, results_type='inputs')
# can be used as the run_params input here.
def get_transition_graphs(
    run_params,
    sweep_type,
    mdps_folder=data.MDPS_FOLDER,
    policies_folder=data.POLICIES_FOLDER
):
    if sweep_type == 'single_agent':
        return [data.load_graph_from_dot_file(run_params.get('mdp_graph'), folder=mdps_folder)]
    
    elif sweep_type == 'multiagent_fixed_policy':
        return [
            data.load_graph_from_dot_file(
                run_params.get('mdp_graph_agent_A'), folder=mdps_folder
            ),
            data.load_graph_from_dot_file(
                run_params.get('policy_graph_agent_B'), folder=policies_folder
            ),
            data.load_graph_from_dot_file(
                run_params.get('mdp_graph_agent_B'), folder=mdps_folder
            )
        ]
    
    elif sweep_type == 'multiagent_with_reward':
        return [
            data.load_graph_from_dot_file(
                run_params.get('mdp_graph_agent_A'), folder=mdps_folder
            ),
            data.load_graph_from_dot_file(
                run_params.get('seed_policy_graph_agent_B', None), folder=policies_folder
            ),
            data.load_graph_from_dot_file(
                run_params.get('mdp_graph_agent_B'), folder=mdps_folder
            )
        ]

def get_properties_from_run(
    sweep_id,
    run_suffix='',
    mdps_folder=data.MDPS_FOLDER,
    policies_folder=data.POLICIES_FOLDER
):
    inputs = get_sweep_run_results(sweep_id, run_suffix, results_type='inputs')
    outputs = get_sweep_run_results(sweep_id, run_suffix, results_type='outputs')
    diagnostics = get_sweep_run_results(sweep_id, run_suffix, results_type='diagnostics')
    sweep_type = misc.determine_sweep_type(inputs)

    extra_props = {
        'reward_samples_agent_B': outputs['reward_samples_agent_B'],
        'power_samples_agent_B': outputs['power_samples_agent_B'],
        'reward_correlation': inputs['reward_correlation'],
        'discount_rate_agent_B': inputs['discount_rate_agent_B']
    } if sweep_type == 'multiagent_with_reward' else {}

    return {
        'reward_samples': outputs['reward_samples'],
        'power_samples': outputs['power_samples'],
        'discount_rate': inputs['discount_rate'],
        'convergence_threshold': inputs['convergence_threshold'],
        'sweep_type': sweep_type,
        'transition_graphs': get_transition_graphs(
            inputs, sweep_type, mdps_folder=mdps_folder, policies_folder=policies_folder
        ),
        **extra_props,
        **diagnostics
    }