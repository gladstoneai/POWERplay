from .lib.utils import dist
from .lib.utils import graph
from .lib.utils import misc
from .lib import data
from .lib import get
from .lib import check
from . import mdp
from . import policy
from . import multi
from . import view
from . import anim
from . import test
from . import launch
from . import rep
from . import ui
from . import viz

# 1) TESTING & REPLICATION

def test_vanilla():
    test.test_vanilla()

def test_gridworld():
    test.test_gridworld()

def test_stochastic():
    test.test_stochastic()

def test_multiagent():
    test.test_multiagent()

def test_reward_correlation():
    test.test_reward_correlation()

def run_all_tests():
    test_vanilla()
    test_gridworld()
    test_stochastic()
    test_multiagent()
    test_reward_correlation()

def reproduce_figure(post_number, fig_number):

    if post_number == 1 and fig_number == 1:
        rep.part_1_fig_1()
    
    elif post_number == 1 and fig_number in [2, 3, 4]:
        rep.part_1_fig_2_3_4()
    
    elif post_number == 1 and fig_number in [5, 6]:
        rep.part_1_fig_5_6()
    
    elif post_number == 2 and fig_number == 1:
        rep.part_2_fig_1()
    
    elif post_number == 2 and fig_number == 5:
        rep.part_2_fig_5()
    
    elif post_number == 2 and fig_number in [2, 3, 4, 6, 7, 8, 9, 10]:
        rep.part_2_fig_2_3_4_6_7_8_9_10()
    
    elif post_number == 3 and fig_number in [2, 3, 4]:
        rep.part_3_fig_2_3_4()
    
    elif post_number == 3 and fig_number in [5, 6, 7]:
        rep.part_3_fig_5_6_7()
    
    elif post_number == 3 and fig_number in [8, 9]:
        rep.part_3_fig_8_9()
    
    elif post_number == 3 and fig_number in [10, 11]:
        rep.part_3_fig_10_11()
    
    else:
        print('Invalid post_number and fig_number combination.')

# 2) SETTING UP THE ENVIRONMENT

def construct_single_agent_gridworld_mdp(
    num_rows,
    num_cols,
    squares_to_delete=[],
    stochastic_noise_level=0,
    noise_bias={},
    description='single-agent gridworld'
):
    return mdp.gridworld_to_stochastic_graph(
            mdp.construct_gridworld(
                num_rows, num_cols, name=description, squares_to_delete=squares_to_delete
            ),
            stochastic_noise_level=stochastic_noise_level,
            noise_bias=noise_bias
        )

def construct_multiagent_gridworld_mdp(
    num_rows,
    num_cols,
    squares_to_delete=[],
    description='multi-agent gridworld'
):
    return multi.create_joint_multiagent_graph(
        construct_single_agent_gridworld_mdp(
            num_rows,
            num_cols,
            squares_to_delete=squares_to_delete,
            stochastic_noise_level=0,
            noise_bias={},
            description=description
        )
    )

def edit_gridworld_mdp(mdp_graph):
    return ui.update_mdp_graph_interface(mdp_graph)

def construct_policy_from_mdp(mdp_graph, acting_agent_is_H=False):
    return policy.quick_mdp_to_policy(mdp_graph, acting_agent_is_H=acting_agent_is_H)

def edit_gridworld_policy(policy_graph):
    return ui.update_multiagent_policy_interface(policy_graph)

def single_agent_to_multiagent_policy(single_agent_policy_graph, acting_agent_is_H=False):
    return policy.single_agent_to_multiagent_policy_graph(
        single_agent_policy_graph, acting_agent_is_H=acting_agent_is_H
    )

def view_gridworld(gridworld_mdp_graph):
    view.view_gridworld(gridworld_mdp_graph)

def plot_mdp_or_policy(
    mdp_or_policy_graph,
    subgraphs_per_row=4,
    subgraphs_per_figure=128
):
    viz.plot_mdp_or_policy(
        mdp_or_policy_graph,
        show=True,
        subgraphs_per_row=subgraphs_per_row,
        number_of_states_per_figure=subgraphs_per_figure,
        reward_to_plot=None,
        save_handle='temp',
        graph_name='graph',
        save_folder=data.TEMP_FOLDER,
        temp_folder=data.TEMP_FOLDER
    )

def save_mdp_graph(mdp_graph, mdp_filename):
    data.save_graph_to_dot_file(mdp_graph, mdp_filename, folder=data.MDPS_FOLDER)

def load_mdp_graph(mdp_filename):
    data.load_graph_from_dot_file(mdp_filename, folder=data.MDPS_FOLDER)

def save_policy_graph(policy_graph, policy_filename):
    data.save_graph_to_dot_file(policy_graph, policy_filename, folder=data.POLICIES_FOLDER)

def load_policy_graph(policy_filename):
    data.load_graph_from_dot_file(policy_filename, folder=data.POLICIES_FOLDER)

# 3) LAUNCHING EXPERIMENTS

def launch_experiment(
    config_filename,
    wandb_entity=data.get_settings_value(data.WANDB_ENTITY_PATH, settings_filename=data.SETTINGS_FILENAME),
    wandb_project='uncategorized',
    number_of_workers=None,
    plot_as_gridworld=False,
    plot_distributions=False,
    plot_correlations=False,
    diagnostic_mode=True
):
    launch.launch_sweep(
        config_filename,
        sweep_local_id=misc.generate_sweep_id(),
        entity=wandb_entity,
        wandb_api_key=data.get_settings_value(data.WANDB_KEY_PATH, settings_filename=data.SETTINGS_FILENAME),
        project=wandb_project,
        sweep_config_folder=data.SWEEP_CONFIGS_FOLDER,
        output_folder_local=data.EXPERIMENT_FOLDER,
        number_of_workers=number_of_workers,
        plot_as_gridworld=plot_as_gridworld,
        plot_distributions=plot_distributions,
        plot_correlations=plot_correlations,
        diagnostic_mode=diagnostic_mode,
        plot_in_log_scale=False,
        force_basic_font=False
    )

# 4) VISUALIZING RESULTS

def visualize_correlated_reward_samples(
    num_samples,
    distribution_config={ 'dist_name': 'uniform', 'params': [0, 1] },
    correlation=0,
    symmetric_interval=None,
    random_seed=0,
    fig_name=None
):
    view.plot_correlated_reward_samples(
        num_samples,
        distribution_config=distribution_config,
        correlation=correlation,
        symmetric_interval=symmetric_interval,
        distribution_dict=dist.DISTRIBUTION_DICT,
        random_seed=random_seed,
        show=True,
        fig_name=fig_name,
        save_folder=data.TEMP_FOLDER
    )

def visualize_all_correlated_reward_samples(
    num_samples,
    distribution_config={ 'dist_name': 'uniform', 'params': [0, 1] },
    correlations_list=[],
    symmetric_interval=None,
    random_seed=0,
    fig_name='correlated_reward_animation'
):
    fig_filenames_ = []

    for correlation in correlations_list:
        fig_filename = '{0}-correlation_{1}'.format(fig_name, correlation)

        view.plot_correlated_reward_samples(
            num_samples,
            distribution_config=distribution_config,
            correlation=correlation,
            symmetric_interval=symmetric_interval,
            distribution_dict=dist.DISTRIBUTION_DICT,
            random_seed=random_seed,
            xlim=None,
            ylim=None,
            show=False,
            fig_name=fig_filename,
            save_folder=data.TEMP_FOLDER
        )

        fig_filenames_ += [fig_filename]

    anim.animate_from_filenames(
        fig_filenames_,
        fig_name,
        ms_per_frame=100,
        frames_at_start=8,
        frames_at_end=4,
        input_folder_or_list=data.TEMP_FOLDER,
        output_folder=data.TEMP_FOLDER
    )

def visualize_alignment_plots(
    sweep_id,
    include_baseline_powers=True,
    fig_name='alignment_plot_animation'
):
    view.plot_specific_power_alignments(
        sweep_id,
        show=False,
        ms_per_frame=100,
        graph_padding_x=0.05,
        graph_padding_y=0.05,
        frames_at_start=8,
        frames_at_end=4,
        fig_name=fig_name,
        include_zero_line=False,
        include_baseline_powers=include_baseline_powers,
        data_folder=data.EXPERIMENT_FOLDER,
        save_folder=data.TEMP_FOLDER
    )

def visualize_variable_vs_power_correlations(
    x_axis_values,
    sweep_ids_list,
    show=True,
    fig_name='temp',
    x_axis_label='name of variable',
    save_folder=data.TEMP_FOLDER
):
    power_correlation_data = [
        get.get_reward_correlations_and_powers_from_sweep(sweep_id, include_baseline_power=False) for (
            sweep_id
        ) in sweep_ids_list
    ]

    all_powers_H_ = []
    all_powers_A_ = []

    for correlation_item in power_correlation_data:
        all_powers_H_ += correlation_item['all_powers_H']
        all_powers_A_ += correlation_item['all_powers_A']
    
    view.plot_power_correlation_relationship(
        x_axis_values,
        all_powers_H_,
        all_powers_A_,
        x_axis_label=x_axis_label,
        show=show,
        fig_name=fig_name,
        save_folder=save_folder
    )

def visualize_average_powers(
    sweep_id,
    include_baseline_power=True,
    fig_name=None
):
    view.plot_alignment_curves(
        sweep_id,
        show=True,
        plot_title='Mean POWERs',
        include_baseline_power=include_baseline_power,
        fig_name=fig_name,
        y_axis_bounds=None,
        data_folder=data.EXPERIMENT_FOLDER,
        save_folder=data.TEMP_FOLDER
    )

def visualize_full_gridworld_rollout(
    sweep_id,
    run_suffix='',
    initial_state='(0, 0)_H^(0, 0)_A',
    reward_sample_index=0,
    number_of_steps=20,
    agent_whose_rewards_are_displayed='H',
    fig_name='gridworld_rollout'
):
    check.check_agent_label(agent_whose_rewards_are_displayed)

    run_properties = get.get_properties_from_run(sweep_id, run_suffix=run_suffix)
    policy_data = policy.sample_optimal_policy_data_from_run(run_properties, reward_sample_index=reward_sample_index)

    view.plot_gridworld_rollout(
        graph.get_states_from_graph(policy_data['policy_graph_H']),
        policy.simulate_policy_rollout(
            initial_state,
            policy_data['policy_graph_H'],
            policy_data['mdp_graph'],
            policy_graph_A=policy_data['policy_graph_A'],
            number_of_steps=number_of_steps,
            random_seed=0
        ),
        reward_function=policy_data['reward_function_{}'.format(agent_whose_rewards_are_displayed)],
        ms_per_frame=400,
        show=False,
        agent_whose_rewards_are_displayed=agent_whose_rewards_are_displayed,
        save_handle=fig_name,
        save_folder=data.TEMP_FOLDER
    )

def generate_sweep_animation(
    sweep_id,
    animation_param,
    figure_prefix=None
):
    anim.generate_sweep_animations(
        sweep_id,
        animation_param=animation_param,
        figure_prefix=figure_prefix,
        ms_per_frame=100,
        frames_at_start=4,
        frames_at_end=8,
        experiment_folder=data.EXPERIMENT_FOLDER
    )
