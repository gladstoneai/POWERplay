import matplotlib.pyplot as plt

from .lib.utils import render
from .lib.utils import graph
from .lib import data
from . import anim
from . import viz
from . import policy

def plot_gridworld_rollout(
    state_list,
    state_rollout,
    reward_function=None,
    show=True,
    agent_whose_rewards_are_displayed='A',
    save_handle='gridworld_rollout',
    save_folder=data.TEMP_FOLDER
):
    all_figs_ = []
    for i in range(len(state_rollout)):
        all_figs_ += [render.render_gridworld_rollout_snapshot(
            state_list,
            state_rollout[i],
            reward_function=reward_function,
            agent_whose_rewards_are_displayed=agent_whose_rewards_are_displayed
        )]
        data.save_figure(all_figs_[-1], '{0}-{1}'.format(save_handle, i), folder=save_folder)

    anim.animate_rollout(
        save_handle,
        len(state_rollout),
        output_filename='{}-animation'.format(save_handle),
        input_folder=save_folder,
        output_folder=save_folder
    )

    if show:
        plt.show()

    return all_figs_

# TODO: Document.
def plot_policy_sample(
    policy_graph,
    reward_function,
    discount_rate,
    show=True,
    subgraphs_per_row=4,
    number_of_states_per_figure=128,
    save_handle='temp',
    save_folder=data.TEMP_FOLDER,
    temp_folder=data.TEMP_FOLDER,
):
    return viz.plot_mdp_or_policy(
        policy_graph,
        show=show,
        subgraphs_per_row=subgraphs_per_row,
        number_of_states_per_figure=number_of_states_per_figure,
        reward_to_plot=reward_function,
        discount_rate_to_plot=discount_rate,
        save_handle=save_handle,
        graph_name='policy_reward_sample',
        save_folder=save_folder,
        temp_folder=temp_folder
    )

# NOTE: This function is not optimized for speed and runs very slowly, even for just 10 rollouts.
# It's also not capable of saving the plot and just shows it.
# run_properties: outupt of get.get_properties_from_run(sweep_id, run_suffix=run_suffix)
def plot_rollout_powers(run_properties, initial_state=None, number_of_rollouts=10, number_of_steps=20):
    state_list = graph.get_states_from_graph(run_properties['transition_graphs'][0])
    powers_A = run_properties['power_samples'].mean(dim=0)

    all_power_plot_rollouts_ = []

    for reward_sample_index in range(number_of_rollouts):

        print(reward_sample_index)
        policy_data = policy.sample_optimal_policy_from_run(run_properties, reward_sample_index=reward_sample_index)

        rollout = policy.simulate_policy_rollout(
            initial_state,
            policy_data['policy_graph_A'],
            policy_data['mdp_graph_A'],
            policy_graph_B=policy_data['policy_graph_B'],
            mdp_graph_B=policy_data['mdp_graph_B'],
            number_of_steps=number_of_steps
        )

        power_rollout = [powers_A[state_list.index(state)].item() for state in rollout]
        # In either multiagent case, we have to skip the rollout steps where Agent B moves
        power_plot_rollout = power_rollout if run_properties['sweep_type'] == 'single_agent' else power_rollout[::2]

        plt.plot(list(range(number_of_steps + 1)), power_plot_rollout, 'b-')
        all_power_plot_rollouts_ += [power_plot_rollout]

    plt.show()

    return all_power_plot_rollouts_