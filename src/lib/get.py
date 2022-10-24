from .utils import graph
from .utils import misc
from . import data
from . import check

def get_sweep_run_suffixes_for_param(sweep_id, param_name, folder=data.EXPERIMENT_FOLDER):
    sweep_name = data.get_full_sweep_name_from_id(sweep_id, folder=folder)
    sweep_params = data.load_full_sweep(sweep_name, folder=folder)['parameters']

    try:
        run_suffixes_ = sorted([
            misc.build_run_suffix(
                { param_name: sweep_params[param_name]['values'][i] }, [param_name]
            ) for i in range(len(sweep_params[param_name]['values']))
        ])
    
    except KeyError:
        run_suffixes_ = ['']
    
    return run_suffixes_

def get_sweep_run_results(sweep_id, run_suffix, results_type='outputs', folder=data.EXPERIMENT_FOLDER):
    sweep_name = data.get_full_sweep_name_from_id(sweep_id, folder=folder)

    return data.load_full_sweep(
        sweep_name, folder=folder
    )['all_runs_data']['{0}-{1}'.format(sweep_name, run_suffix)].get(results_type, {})

def get_sweep_type(sweep_id, folder=data.EXPERIMENT_FOLDER):
    return misc.determine_sweep_type(
        data.load_full_sweep(data.get_full_sweep_name_from_id(sweep_id, folder=folder), folder=folder)['parameters']
    )

def get_sweep_state_list(sweep_id, folder=data.EXPERIMENT_FOLDER):
    parameters = data.load_full_sweep(
        data.get_full_sweep_name_from_id(sweep_id, folder=folder), folder=folder
    )['parameters']
    mdp_graph_param = parameters.get('mdp_graph', parameters.get('joint_mdp_graph'))

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

def get_transition_graphs(
    run_params,
    sweep_type,
    check_graph_compatibilities=True,
    mdps_folder=data.MDPS_FOLDER,
    policies_folder=data.POLICIES_FOLDER
):
    if sweep_type == 'single_agent':
        return [data.load_graph_from_dot_file(run_params['mdp_graph'], folder=mdps_folder)]
    
    elif sweep_type == 'multiagent_fixed_policy':
        joint_mdp_graph = data.load_graph_from_dot_file(run_params['joint_mdp_graph'], folder=mdps_folder)
        policy_graph_A = data.load_graph_from_dot_file(run_params['policy_graph_agent_A'], folder=policies_folder)

        if check_graph_compatibilities:
            check.check_joint_mdp_and_policy_compatibility(
                joint_mdp_graph, policy_graph_A, policy_is_for_agent_H=False
            )

        return [joint_mdp_graph, policy_graph_A]
    
    elif sweep_type == 'multiagent_with_reward':
        joint_mdp_graph = data.load_graph_from_dot_file(run_params['joint_mdp_graph'], folder=mdps_folder)
        seed_policy_graph_A = data.load_graph_from_dot_file(run_params['seed_policy_graph_agent_A'], folder=policies_folder)

        if check_graph_compatibilities:
            check.check_joint_mdp_and_policy_compatibility(
                joint_mdp_graph, seed_policy_graph_A, policy_is_for_agent_H=False
            )

        return [joint_mdp_graph, seed_policy_graph_A]

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
        'reward_samples_agent_A': outputs['reward_samples_agent_A'],
        'power_samples_agent_A': outputs['power_samples_agent_A'],
        'reward_correlation': inputs['reward_correlation'],
        'discount_rate_agent_A': inputs['discount_rate_agent_A']
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

def get_reward_correlations_and_powers_from_sweep(sweep_id, include_baseline_power=True, folder=data.EXPERIMENT_FOLDER):
    run_suffixes = get_sweep_run_suffixes_for_param(sweep_id, 'reward_correlation', folder=folder)

    all_run_props_ = []

    print()

    for run_suffix in run_suffixes:

        print('Accessing run {}...'.format(run_suffix))

        all_run_props_ += [get_properties_from_run(sweep_id, run_suffix=run_suffix)]
    
    print()

    return {
        'reward_correlations': [run_props['reward_correlation'] for run_props in all_run_props_],
        'all_powers_H': [run_props['power_samples'].mean(dim=0) for run_props in all_run_props_],
        'all_powers_A': [run_props['power_samples_agent_A'].mean(dim=0) for run_props in all_run_props_],
        **(
            {
                'baseline_powers_H': all_run_props_[0]['power_samples_H_against_seed'].mean(dim=0)
            } if include_baseline_power else {}
        )
    }
