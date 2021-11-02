import matplotlib.pyplot as plt
import numpy as np

def policy_rollout(policy, reward_function, state_list, starting_state, number_of_steps):
    if len(policy) != len(reward_function):
        raise Exception('The policy must have the same number of rows as the reward_function has entries.')
    
    if len(reward_function) != len(state_list):
        raise Exception('The reward_function and state_list must have the same length.')   

    state_index = state_list.index(starting_state)

    state_history = [starting_state]
    reward_history = [reward_function[state_index]]

    for _ in range(number_of_steps):
        state_index = np.where(policy[state_index] == 1)[0][0]

        state_history += [state_list[state_index]]
        reward_history += [reward_function[state_index]]
    
    return (
        state_history, reward_history
    )

def plot_power_means(power_distributions, state_list):
    _, ax = plt.subplots()

    ax.bar(
        range(len(state_list)),
        np.mean(power_distributions, axis=0),
        yerr=np.std(power_distributions, axis=0) / np.sqrt(len(power_distributions)),
        align='center',
        ecolor='black',
        capsize=10
    )
    ax.set_ylabel('POWER of state (reward units)')
    ax.set_xticks(range(len(state_list)))
    ax.set_xticklabels(state_list)
    ax.set_title('POWER (mean $\pm$ standard error of the mean) for each state')

    plt.tight_layout()
    plt.show()

def plot_power_distributions(power_distributions, state_list, states_to_plot=None, number_of_bins=30):
    # The terminal state (last in the list) has power 0 in all samples, so we don't plot it by default.
    states_to_plot = state_list[:-1] if (states_to_plot is None) else states_to_plot

    state_indices = [state_list.index(state_label) for state_label in states_to_plot]

    fig, axs = plt.subplots(1, len(state_indices), sharex=True, sharey=True, tight_layout=True)

    for i in range(len(state_indices)):
        axs[i].hist(np.transpose(power_distributions)[state_indices[i]], bins=number_of_bins)
        axs[i].title.set_text(states_to_plot[i])
    
    fig.text(0.5, 0.01, 'POWER sample (reward units)')

    plt.show()

def plot_power_correlations(power_distributions, state_list, state_x, state_list_y=None, number_of_bins=30):
    # The terminal state (last in the list) has power 0 in all samples, so we don't plot it by default.
    state_list_y = state_list[:-1] if (state_list_y is None) else state_list_y

    state_x_index = state_list.index(state_x)
    state_y_indices = [state_list.index(state_label) for state_label in state_list_y]

    fig, axs = plt.subplots(1, len(state_y_indices), sharex=True, sharey=False, tight_layout=True)

    for i in range(len(state_y_indices)):
        axs[i].hist2d(
            np.transpose(power_distributions)[state_x_index],
            np.transpose(power_distributions)[state_y_indices[i]],
            bins=number_of_bins
        )
        axs[i].set_ylabel('POWER sample of state {} (reward units)'.format(state_list_y[i]))

    fig.text(0.5, 0.01, 'POWER sample of state {} (reward units)'.format(state_x))

    plt.show()