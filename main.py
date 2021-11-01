import numpy as np

import lib
import viz
import data

def test():
    discount_rate = 0.9
    power, power_distributions = lib.calculate_power(
        data.ADJACENCY_MATRIX, discount_rate
    )
    viz.plot_power_means(power_distributions, data.STATE_LIST)

    return (
        power,
        power_distributions
    )


# TODO: Build support for different reward distributions.
# TODO? Use multiprocessing to speed up reward sampling.
