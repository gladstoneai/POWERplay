import lib
import data

def test():
    return lib.run_one_experiment(
        adjacency_matrix=data.ADJACENCY_MATRIX,
        discount_rate=0.9,
        save_experiment=False,
        plot_when_done=True,
        state_list=data.STATE_LIST
    )

# TODO: Remove policy iteration.
# TODO? Use multiprocessing to speed up reward sampling.
