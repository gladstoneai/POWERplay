import launch
import data
import check

def test():
    launch.launch_sweep(
        'test_sweep.yaml',
        entity=data.get_settings_value('public.WANDB_DEFAULT_ENTITY'),
        project='uncategorized'
    )

# Ensure all our adjacency matrices pass sanity checks on startup.
# TODO: Save these in dill by default and sanitize on saving rather than on loading.

if __name__ == '__main__':
    for mdp_key, _ in data.MDP_GRAPH_DICT.items():
         check.check_mdp_graph(mdp_key, state_list=data.STATE_LIST, mdp_dict=data.MDP_GRAPH_DICT)



# TODO: Build state_list into the program itself since we shouldn't assume a list format.
# TODO: Test setting random seed.
# TODO: Save with wandb instead of homespun data methods.
# TODO: Investigate writing type hints.
# TODO: Add argparse.ArgumentParser().
# parser = argparse.ArgumentParser()
# parser.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',
#                      help='input batch size for training (default: 8)')
# args = parser.parse_args()
# TODO: Refactor experiment wrapper with https://hydra.cc/ when I understand what experiment configs
# I commonly use.
# TODO: Add ability to run sweep without wandb server access (i.e., offline mode). May be impossible, but
# would be great as it would allow me to run local tests without consuming bandwidth, etc.


# TODO NEXT: Use networkx to construct graph rather than manually writing the adjacency matrix.
# Refactor this when we start investigating new topologies.
# TODO NEXT: Add animations for sweeps.
'''
launch.launch_sweep(
    'sweep-REWARD_ANNEALING_on_ℓ_↙-gamma_0p9-samples_100k.yaml',
    entity=data.get_settings_value('public.WANDB_COLLAB_ENTITY'),
    project='power-project'
)
'''
'''
import networkx as nx
import matplotlib.pyplot as plt

a = nx.DiGraph([
    ('★', '∅'), ('★', 'ℓ_◁'), ('★', 'r_▷'),
    ('∅', '∅'),
    ('ℓ_◁', 'ℓ_↖'), ('ℓ_◁', 'ℓ_↙'),
    ('ℓ_↖', 'ℓ_↙'), ('ℓ_↖', 'TERMINAL'),
    ('ℓ_↙', 'ℓ_↖'), ('ℓ_↙', 'ℓ_↙'),
    ('TERMINAL', 'TERMINAL'),
    ('r_▷', 'r_↗'), ('r_▷', 'r_↘'),
    ('r_↘', 'r_↘'), ('r_↘', 'r_↗'),
    ('r_↗', 'r_↗'), ('r_↗', 'r_↘')
])

ax = plt.subplot()
nx.draw_planar(a, with_labels=True, node_size=300, arrowstyle='->', width='2', font_size=8, font_weight='bold')
plt.show()
'''
expt = test()