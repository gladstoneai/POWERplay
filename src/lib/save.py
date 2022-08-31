from .utils import graph
from . import data

################################################################################

ALL_GRAPH_DESCRIPTIONS = {
    'mdp_graph': 'MDP graph',
    'mdp_graph_A': 'MDP graph for agent A',
    'mdp_graph_B': 'MDP graph for agent B',
    'policy_graph_B': 'Policy graph for agent B',
    'seed_policy_graph_B': 'Seed policy graph for agent B'
}

################################################################################

def save_graphs_and_generate_data(
    input_graphs_dict,
    run_name,
    save_folder=data.TEMP_FOLDER,
    graph_descriptions=ALL_GRAPH_DESCRIPTIONS
):
    for input_graph_name, input_graph in input_graphs_dict.items():
        data.save_graph_to_dot_file(
            input_graph, '{0}-{1}'.format(input_graph_name, run_name), folder=save_folder
        )

    return [{
        'graph_name': input_graph_name,
        'graph_description': graph_descriptions[input_graph_name],
        'graph_data': input_graphs_dict[input_graph_name]
    } for input_graph_name in input_graphs_dict.keys()]

def save_graphs_and_generate_data_from_sweep_type(
    transition_graphs,
    run_name,
    sweep_type,
    save_folder=data.TEMP_FOLDER,
    graph_descriptions=ALL_GRAPH_DESCRIPTIONS
):
    if sweep_type == 'single_agent':
        graph_plotting_data = save_graphs_and_generate_data(
            {
                'mdp_graph': transition_graphs[0]
            },
            run_name,
            save_folder=save_folder,
            graph_descriptions=graph_descriptions
        )

    elif sweep_type == 'multiagent_fixed_policy':
        graph_plotting_data = save_graphs_and_generate_data(
            {
                'mdp_graph': transition_graphs[0],
                'policy_graph_B': transition_graphs[1]
            },
            run_name,
            save_folder=save_folder,
            graph_descriptions=graph_descriptions
        )

    elif sweep_type == 'multiagent_with_reward':
        graph_plotting_data = save_graphs_and_generate_data(
            {
                'mdp_graph_A': transition_graphs[0],
                'seed_policy_graph_B': transition_graphs[1],
                'mdp_graph_B': transition_graphs[2]
            },
            run_name,
            save_folder=save_folder,
            graph_descriptions=graph_descriptions
        )
        
    return [
        graph.get_states_from_graph(transition_graphs[0]),
        graph_plotting_data
    ]