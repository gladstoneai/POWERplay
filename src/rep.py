import pathlib as path
import multiprocessing as mps

from .lib.utils import misc
from .lib import data
from .lib import get
from . import launch
from . import viz
from . import anim
from . import view

def part_1_fig_1():
    sweep_id = misc.generate_sweep_id()
    fig_name = 'PART_1-FIGURE_1'

    launch.launch_sweep(
        '{}.yaml'.format(fig_name),
        sweep_local_id=sweep_id,
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        number_of_workers=mps.cpu_count(),
        plot_distributions=False,
        plot_correlations=False,
        plot_as_gridworld=True,
        project='uncategorized',
        sweep_config_folder=path.Path()/data.SWEEP_CONFIGS_FOLDER/'replication'
    )

    viz.plot_sample_aggregations(
        get.get_properties_from_run(sweep_id)['power_samples'],
        get.get_sweep_state_list(sweep_id),
        show=False,
        plot_as_gridworld=True,
        save_handle=fig_name,
        save_folder=data.TEMP_FOLDER
    )

    print('Figure available in {0}/POWER_means-{1}.png'.format(data.TEMP_FOLDER, fig_name))

def part_1_fig_2_3_4():
    sweep_id = misc.generate_sweep_id()
    fig_name = 'PART_1-FIGURES_2_3_4'

    launch.launch_sweep(
        '{}.yaml'.format(fig_name),
        sweep_local_id=sweep_id,
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        number_of_workers=mps.cpu_count(),
        plot_distributions=False,
        plot_correlations=False,
        plot_as_gridworld=True,
        project='uncategorized',
        sweep_config_folder=path.Path()/data.SWEEP_CONFIGS_FOLDER/'replication'
    )

    # FIGURE 1-2

    save_handle_2 = 'PART_1-FIGURE_2'

    viz.plot_sample_aggregations(
        get.get_properties_from_run(sweep_id, run_suffix='discount_rate__0p01')['power_samples'],
        get.get_sweep_state_list(sweep_id),
        show=False,
        plot_as_gridworld=True,
        save_handle=save_handle_2,
        save_folder=data.TEMP_FOLDER
    )

    print('Figure available in {0}/POWER_means-{1}.png'.format(data.TEMP_FOLDER, save_handle_2))

    # FIGURE 1-3

    save_handle_3 = 'PART_1-FIGURE_3'

    viz.plot_sample_aggregations(
        get.get_properties_from_run(sweep_id, run_suffix='discount_rate__0p99')['power_samples'],
        get.get_sweep_state_list(sweep_id),
        show=False,
        plot_as_gridworld=True,
        save_handle=save_handle_3,
        save_folder=data.TEMP_FOLDER
    )

    print('Figure available in {0}/POWER_means-{1}.png'.format(data.TEMP_FOLDER, save_handle_3))

    # FIGURE 1-4

    save_handle_4 = 'PART_1-FIGURE_4'

    run_suffixes = get.get_sweep_run_suffixes_for_param(sweep_id, 'discount_rate')
    list_of_filenames = ['{0}-{1}'.format(save_handle_4, run_suffix) for run_suffix in run_suffixes]

    for run_suffix, filename in zip(run_suffixes, list_of_filenames):
        viz.plot_sample_aggregations(
            get.get_properties_from_run(sweep_id, run_suffix=run_suffix)['power_samples'],
            get.get_sweep_state_list(sweep_id),
            show=False,
            plot_as_gridworld=True,
            save_handle=filename,
            manual_title='',
            save_folder=data.TEMP_FOLDER
        )

    animation_filenames = ['POWER_means-{}'.format(filename) for filename in list_of_filenames]
    
    anim.animate_from_filenames(
        animation_filenames, save_handle_4, ms_per_frame=100, frames_at_start=4, frames_at_end=8
    )

    print('Figure available in {0}/{1}.gif'.format(data.TEMP_FOLDER, save_handle_4))

def part_1_fig_5_6():
    sweep_id = misc.generate_sweep_id()
    fig_name = 'PART_1-FIGURES_5_6'

    launch.launch_sweep(
        '{}.yaml'.format(fig_name),
        sweep_local_id=sweep_id,
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        number_of_workers=mps.cpu_count(),
        plot_distributions=False,
        plot_correlations=False,
        plot_as_gridworld=True,
        project='uncategorized',
        sweep_config_folder=path.Path()/data.SWEEP_CONFIGS_FOLDER/'replication'
    )

    # FIGURE 1-5

    save_handle_5 = 'PART_1-FIGURE_5'

    viz.plot_sample_aggregations(
        get.get_properties_from_run(sweep_id, run_suffix='discount_rate__0p1')['power_samples'],
        get.get_sweep_state_list(sweep_id),
        show=False,
        plot_as_gridworld=True,
        save_handle=save_handle_5,
        save_folder=data.TEMP_FOLDER
    )

    print('Figure available in {0}/POWER_means-{1}.png'.format(data.TEMP_FOLDER, save_handle_5))

    # FIGURE 1-6

    save_handle_6 = 'PART_1-FIGURE_6'

    viz.plot_sample_aggregations(
        get.get_properties_from_run(sweep_id, run_suffix='discount_rate__0p99')['power_samples'],
        get.get_sweep_state_list(sweep_id),
        show=False,
        plot_as_gridworld=True,
        save_handle=save_handle_6,
        save_folder=data.TEMP_FOLDER,
        manual_title=''
    )

    print('Figure available in {0}/POWER_means-{1}.png'.format(data.TEMP_FOLDER, save_handle_6))

def part_2_fig_1():
    fig_name = 'PART_2-FIGURE_1'

    view.plot_correlated_reward_samples(
        256,
        distribution_config={ 'dist_name': 'uniform', 'params': [0, 1] },
        correlation=1,
        show=False,
        fig_name=fig_name,
    )

    print('Figure available in {0}/{1}.png'.format(data.TEMP_FOLDER, fig_name))

def part_2_fig_5():
    fig_name = 'PART_2-FIGURE_5'

    view.plot_correlated_reward_samples(
        256,
        distribution_config={ 'dist_name': 'uniform', 'params': [0, 1] },
        correlation=0,
        show=False,
        fig_name=fig_name,
    )

    print('Figure available in {0}/{1}.png'.format(data.TEMP_FOLDER, fig_name))

def part_2_fig_2_3_4_6_7_8_9_10():
    sweep_id = misc.generate_sweep_id()
    fig_name = 'PART_2-FIGURES_2_3_4_6_7_8_9_10'

    launch.launch_sweep(
        '{}.yaml'.format(fig_name),
        sweep_local_id=sweep_id,
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        number_of_workers=mps.cpu_count(),
        plot_distributions=False,
        plot_correlations=False,
        plot_as_gridworld=True,
        project='uncategorized',
        sweep_config_folder=path.Path()/data.SWEEP_CONFIGS_FOLDER/'replication'
    )

    # FIGURE 2-2

    save_handle_2 = 'PART_2-FIGURE_2'

    viz.plot_sample_aggregations(
        get.get_properties_from_run(sweep_id, run_suffix='reward_correlation__1')['power_samples'],
        get.get_sweep_state_list(sweep_id),
        show=False,
        plot_as_gridworld=True,
        save_handle=save_handle_2,
        save_folder=data.TEMP_FOLDER
    )

    print('Figure available in {0}/POWER_means-{1}.png'.format(data.TEMP_FOLDER, save_handle_2))

    # FIGURE 2-3

    save_handle_3 = 'PART_2-FIGURE_3'

    viz.plot_sample_aggregations(
        get.get_properties_from_run(sweep_id, run_suffix='reward_correlation__1')['power_samples_agent_A'],
        get.get_sweep_state_list(sweep_id),
        show=False,
        plot_as_gridworld=True,
        save_handle=save_handle_3,
        save_folder=data.TEMP_FOLDER
    )

    print('Figure available in {0}/POWER_means-{1}.png'.format(data.TEMP_FOLDER, save_handle_3))

    # FIGURES 2-4, 2-8, 2-9 AND 2-10

    save_handle_4_8_9_10 = 'PART_2-FIGURE_4_8_9_10'

    view.plot_specific_power_alignments(
        sweep_id,
        show=False,
        ms_per_frame=100,
        frames_at_start=4,
        frames_at_end=8,
        fig_name=save_handle_4_8_9_10,
        include_baseline_powers=False,
        include_zero_line=True
    )

    print('Figures available in {0}/{1}-[...].png'.format(data.TEMP_FOLDER, save_handle_4_8_9_10))
    print('Figure available in {0}/{1}-animation.gif'.format(data.TEMP_FOLDER, save_handle_4_8_9_10))

    # FIGURE 2-6

    save_handle_6 = 'PART_2-FIGURE_6'

    viz.plot_sample_aggregations(
        get.get_properties_from_run(sweep_id, run_suffix='reward_correlation__0')['power_samples'],
        get.get_sweep_state_list(sweep_id),
        show=False,
        plot_as_gridworld=True,
        save_handle=save_handle_6,
        save_folder=data.TEMP_FOLDER
    )

    print('Figure available in {0}/POWER_means-{1}.png'.format(data.TEMP_FOLDER, save_handle_6))

    # FIGURE 2-7

    save_handle_7 = 'PART_2-FIGURE_7'

    viz.plot_sample_aggregations(
        get.get_properties_from_run(sweep_id, run_suffix='reward_correlation__0')['power_samples_agent_A'],
        get.get_sweep_state_list(sweep_id),
        show=False,
        plot_as_gridworld=True,
        save_handle=save_handle_7,
        save_folder=data.TEMP_FOLDER
    )

    print('Figure available in {0}/POWER_means-{1}.png'.format(data.TEMP_FOLDER, save_handle_7))

    # FIGURE 2-9

    save_handle_9 = 'PART_2-FIGURE_9'

    fig_filenames_ = []

    for correlation in [i/20 for i in range(21)]:
        fig_filename = '{0}-correlation_{1}'.format(save_handle_9, correlation)

        view.plot_correlated_reward_samples(
            256,
            distribution_config={ 'dist_name': 'uniform', 'params': [0, 1] },
            correlation=correlation,
            xlim=[0, 1],
            ylim=[0, 1],
            show=False,
            fig_name=fig_filename,
            save_folder=data.TEMP_FOLDER
        )

        fig_filenames_ += [fig_filename]
    
    anim.animate_from_filenames(
        fig_filenames_,
        save_handle_9,
        ms_per_frame=100,
        frames_at_start=4,
        frames_at_end=8,
        input_folder_or_list=data.TEMP_FOLDER,
        output_folder=data.TEMP_FOLDER
    )

    print('Figure available in {0}/{1}.gif'.format(data.TEMP_FOLDER, save_handle_9))

def part_3_fig_2_3_4():
    sweep_id = misc.generate_sweep_id()
    fig_name = 'PART_3-FIGURES_2_3_4'

    launch.launch_sweep(
        '{}.yaml'.format(fig_name),
        sweep_local_id=sweep_id,
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        number_of_workers=mps.cpu_count(),
        plot_distributions=False,
        plot_correlations=False,
        plot_as_gridworld=True,
        project='uncategorized',
        sweep_config_folder=path.Path()/data.SWEEP_CONFIGS_FOLDER/'replication'
    )

   # FIGURE 3-2

    save_handle_2 = 'PART_3-FIGURE_2'

    viz.plot_sample_aggregations(
        get.get_properties_from_run(sweep_id)['power_samples'],
        get.get_sweep_state_list(sweep_id),
        show=False,
        plot_as_gridworld=True,
        manual_title='',
        save_handle=save_handle_2,
        save_folder=data.TEMP_FOLDER
    )

    print('Figure available in {0}/POWER_means-{1}.png'.format(data.TEMP_FOLDER, save_handle_2))

    # FIGURE 3-3

    save_handle_3 = 'PART_3-FIGURE_3'

    viz.plot_sample_aggregations(
        get.get_properties_from_run(sweep_id)['power_samples_agent_A'],
        get.get_sweep_state_list(sweep_id),
        show=False,
        plot_as_gridworld=True,
        manual_title='',
        save_handle=save_handle_3,
        save_folder=data.TEMP_FOLDER
    )

    print('Figure available in {0}/POWER_means-{1}.png'.format(data.TEMP_FOLDER, save_handle_3))

    # FIGURE 3-4

    save_handle_4 = 'PART_3-FIGURE_4'

    view.plot_specific_power_alignments(
        sweep_id,
        show=False,
        ms_per_frame=100,
        frames_at_start=4,
        frames_at_end=8,
        fig_name=save_handle_4,
        include_baseline_powers=False,
        include_zero_line=True
    )

    print('Figure available in {0}/{1}-[...].png'.format(data.TEMP_FOLDER, save_handle_4))

def part_3_fig_5_6_7():
    sweep_id = misc.generate_sweep_id()
    fig_name = 'PART_3-FIGURES_5_6_7'

    launch.launch_sweep(
        '{}.yaml'.format(fig_name),
        sweep_local_id=sweep_id,
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        number_of_workers=mps.cpu_count(),
        plot_distributions=False,
        plot_correlations=False,
        plot_as_gridworld=True,
        project='uncategorized',
        sweep_config_folder=path.Path()/data.SWEEP_CONFIGS_FOLDER/'replication'
    )

   # FIGURE 3-5

    save_handle_5 = 'PART_3-FIGURE_5'

    viz.plot_sample_aggregations(
        get.get_properties_from_run(sweep_id)['power_samples'],
        get.get_sweep_state_list(sweep_id),
        show=False,
        plot_as_gridworld=True,
        manual_title='',
        save_handle=save_handle_5,
        save_folder=data.TEMP_FOLDER
    )

    print('Figure available in {0}/POWER_means-{1}.png'.format(data.TEMP_FOLDER, save_handle_5))

    # FIGURE 3-6

    save_handle_6 = 'PART_3-FIGURE_6'

    viz.plot_sample_aggregations(
        get.get_properties_from_run(sweep_id)['power_samples_agent_A'],
        get.get_sweep_state_list(sweep_id),
        show=False,
        plot_as_gridworld=True,
        manual_title='',
        save_handle=save_handle_6,
        save_folder=data.TEMP_FOLDER
    )

    print('Figure available in {0}/POWER_means-{1}.png'.format(data.TEMP_FOLDER, save_handle_6))

    # FIGURE 3-7

    save_handle_7 = 'PART_3-FIGURE_7'

    view.plot_specific_power_alignments(
        sweep_id,
        show=False,
        ms_per_frame=100,
        frames_at_start=4,
        frames_at_end=8,
        fig_name=save_handle_7,
        include_baseline_powers=False,
        include_zero_line=True
    )

    print('Figure available in {0}/{1}-[...].png'.format(data.TEMP_FOLDER, save_handle_7))

def part_3_fig_8_9():
    sweep_id = misc.generate_sweep_id()
    fig_name = 'PART_3-FIGURES_8_9'

    launch.launch_sweep(
        '{}.yaml'.format(fig_name),
        sweep_local_id=sweep_id,
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        number_of_workers=mps.cpu_count(),
        plot_distributions=False,
        plot_correlations=False,
        plot_as_gridworld=True,
        project='uncategorized',
        sweep_config_folder=path.Path()/data.SWEEP_CONFIGS_FOLDER/'replication'
    )

   # FIGURE 3-8

    save_handle_8a = 'PART_3-FIGURE_8A'

    viz.plot_sample_aggregations(
        get.get_properties_from_run(sweep_id)['power_samples'],
        get.get_sweep_state_list(sweep_id),
        show=False,
        plot_as_gridworld=True,
        manual_title='',
        save_handle=save_handle_8a,
        save_folder=data.TEMP_FOLDER
    )

    print('Figure available in {0}/POWER_means-{1}.png'.format(data.TEMP_FOLDER, save_handle_8a))

    save_handle_8b = 'PART_3-FIGURE_8B'

    viz.plot_sample_aggregations(
        get.get_properties_from_run(sweep_id)['power_samples_agent_A'],
        get.get_sweep_state_list(sweep_id),
        show=False,
        plot_as_gridworld=True,
        manual_title='',
        save_handle=save_handle_8b,
        save_folder=data.TEMP_FOLDER
    )

    print('Figure available in {0}/POWER_means-{1}.png'.format(data.TEMP_FOLDER, save_handle_8b))

    # FIGURE 3-9

    save_handle_9 = 'PART_3-FIGURE_9'

    view.plot_specific_power_alignments(
        sweep_id,
        show=False,
        ms_per_frame=100,
        frames_at_start=4,
        frames_at_end=8,
        fig_name=save_handle_9,
        include_baseline_powers=False,
        include_zero_line=True,
        graph_padding_x=0.01,
        graph_padding_y=0.01
    )

    print('Figure available in {0}/{1}-[...].png'.format(data.TEMP_FOLDER, save_handle_9))

def part_3_fig_10_11():
    sweep_id = misc.generate_sweep_id()
    fig_name = 'PART_3-FIGURES_10_11'

    launch.launch_sweep(
        '{}.yaml'.format(fig_name),
        sweep_local_id=sweep_id,
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        number_of_workers=mps.cpu_count(),
        plot_distributions=False,
        plot_correlations=False,
        plot_as_gridworld=True,
        project='uncategorized',
        sweep_config_folder=path.Path()/data.SWEEP_CONFIGS_FOLDER/'replication'
    )

   # FIGURE 3-10

    save_handle_10a = 'PART_3-FIGURE_10A'

    viz.plot_sample_aggregations(
        get.get_properties_from_run(sweep_id)['power_samples'],
        get.get_sweep_state_list(sweep_id),
        show=False,
        plot_as_gridworld=True,
        manual_title='',
        save_handle=save_handle_10a,
        save_folder=data.TEMP_FOLDER
    )

    print('Figure available in {0}/POWER_means-{1}.png'.format(data.TEMP_FOLDER, save_handle_10a))

    save_handle_10b = 'PART_3-FIGURE_10B'

    viz.plot_sample_aggregations(
        get.get_properties_from_run(sweep_id)['power_samples_agent_A'],
        get.get_sweep_state_list(sweep_id),
        show=False,
        plot_as_gridworld=True,
        manual_title='',
        save_handle=save_handle_10b,
        save_folder=data.TEMP_FOLDER
    )

    print('Figure available in {0}/POWER_means-{1}.png'.format(data.TEMP_FOLDER, save_handle_10b))

    # FIGURE 3-11

    save_handle_11 = 'PART_3-FIGURE_11'

    view.plot_specific_power_alignments(
        sweep_id,
        show=False,
        ms_per_frame=100,
        frames_at_start=4,
        frames_at_end=8,
        fig_name=save_handle_11,
        include_baseline_powers=False,
        include_zero_line=True,
        graph_padding_x=0.01,
        graph_padding_y=0.01
    )

    print('Figure available in {0}/{1}-[...].png'.format(data.TEMP_FOLDER, save_handle_11))