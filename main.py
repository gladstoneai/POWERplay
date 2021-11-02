import lib
import viz
import data

def test():
    discount_rate = 0.9
    power, power_distributions = lib.calculate_power(
        data.ADJACENCY_MATRIX,
        discount_rate,
        reward_distributions=[
            lambda x: x,
            lambda x: 1,
            lambda x: x**2,
            lambda x: x**3,
            lambda x: 1 - x,
            lambda x: 1,
            lambda x: 2*x + 4,
            lambda x: 1
        ],
        reward_sample_resolution=200
    )
    viz.plot_power_means(power_distributions, data.STATE_LIST)
    viz.plot_power_distributions(power_distributions, data.STATE_LIST)
    viz.plot_power_correlations(power_distributions, data.STATE_LIST, '★')

    return (
        power,
        power_distributions
    )


# TODO: Build an experiment wrapper function.
# TODO? Use multiprocessing to speed up reward sampling.
