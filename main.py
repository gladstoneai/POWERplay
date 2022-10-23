from src.lib.utils import dist

from src import base

# TODO final:
# - Document all base functions
# -- We're using an M1 Mac with 10 cores; may need to change this
# -- base.visualize_alignment_curves
# -- base.visualize_all_alignment_curves
# -- base.visualize_specific_power_alignments
# -- base.visualize_full_gridworld_rollout
# -- base.visualize_correlated_reward_samples
# -- base.visualize_power_relationship_over_multiple_sweeps
# -- base.visualize_all_correlated_reward_samples
# -- base.generate_sweep_animations (wrapper for anim.generate_sweep_animations)
# -- base.plot_policy_sample (wrapper for view.plot_policy_sample)
# -- base.view_gridworld (wrapper for view.view_gridworld)
# - Document all non-base functions
# --- graph: queries MDP and policy graphs
# ---- graph.is_graph_multiagent
# ---- graph.get_states_from_graph
# ---- graph.get_actions_from_graph
# ---- graph.get_unique_single_agent_actions_from_joint_actions
# ---- graph.get_single_agent_actions_from_joint_mdp_graph
# ---- graph.get_available_actions_from_graph_state
# ---- graph.get_available_states_and_probabilities_from_mdp_graph_state_and_action
# ---- graph.graph_to_joint_transition_tensor
# ---- graph.graph_to_full_transition_tensor
# ---- graph.graph_to_policy_tensor
# ---- graph.any_graphs_to_full_transition_tensor
# --- mdp: creates and updates basic MDP graphs
# ---- mdp.add_state_action
# ---- mdp.update_state_action
# ---- mdp.remove_state_action
# ---- mdp.remove_state_completely
# ---- mdp.generate_noised_gridworlds
# --- multi: creates and updates multi-agent graphs
# ---- multi.remove_states_with_overlapping_agents
# ---- multi.create_joint_multiagent_graph
# --- policy: creates and updates policy graphs, manages policy rollouts
# ---- policy.update_state_actions
# ---- policy.single_agent_to_multiagent_policy_graph
# ---- policy.sample_optimal_policy_data_from_run
# ---- policy.simulate_policy_rollout
# ---- policy.policy_tensor_to_graph
# --- data: low-level functions for saving and loading figures, MDP graphs, and experimental results
# ---- data.save_graph_to_dot_file
# ---- data.load_graph_from_dot_file
# --- get: high-level functions for retrieving and preprocessing experimental results
# ---- get.get_properties_from_run
# ---- get.get_sweep_run_suffixes_for_param
# ---- get.get_sweep_run_results
# ---- get.get_sweep_type
# ---- get.get_transition_graphs
# ---- get.get_sweep_state_list
# --- viz: visualizations that get run automatically during experiments
# ---- viz.plot_sample_aggregations
# ---- viz.plot_sample_distributions
# ---- viz.plot_sample_correlations
# --- anim: creates & organizes animations from existing image files
# ---- anim.animate_from_filenames
# --- learn: implementations of RL algorithms for value, policy, and POWER
# ---- learn.value_iteration
# ---- learn.policy_evaluation
# ---- learn.find_optimal_policy
# ---- learn.compute_power_values
# --- runex: runs experiments
# ---- runex.run_one_experiment
# --- dist: low-level functions for building & managing reward distributions
# ---- dist.DISTRIBUTION_DICT
# ---- dist.config_to_pdf_constructor
# ---- dist.config_to_reward_distribution
# ---- dist.generate_correlated_reward_samples
# --- misc: miscellaneous helper functions
# ---- misc.generate_sweep_id
# - Document format for multi-agent graphs, e.g. left_H^stay_A
# - Document the stochastic graph format for MDPs and policies

if __name__ == '__main__':

    pass