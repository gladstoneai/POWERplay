import sys
import torch
import functools as func
import multiprocessing as mps

from .utils import learn
from .utils import misc
from . import check

def output_sample_calculator(
    *args,
    number_of_samples=1,
    iteration_function=learn.value_iteration,
    convergence_threshold=1e-4,
    value_initializations=None,
    worker_pool_size=1,
    worker_id=0
):
    args_by_iteration = [[args[i][j] for i in range(len(args))] for j in range(number_of_samples)]
    all_value_initializations = [None] * number_of_samples if (
        value_initializations is None
    ) else value_initializations

    all_output_samples_ = []

    for j in range(number_of_samples):
        if worker_id == 0: # Only the first worker prints so the pool isn't slowed
            sys.stdout.write('Running samples {0} / {1}'.format(
                worker_pool_size * (j + 1), worker_pool_size * number_of_samples
            ))
            sys.stdout.flush()
            sys.stdout.write('\r')
            sys.stdout.flush()

        all_output_samples_ += [
            iteration_function(
                *args_by_iteration[j],
                value_initialization=all_value_initializations[j],
                convergence_threshold=convergence_threshold
            )
        ]

    if worker_id == 0:
        print() # Jump to newline after stdout.flush()

    return torch.stack(all_output_samples_)

def output_sample_calculator_mps(worker_id, *args, **kwargs):
    return output_sample_calculator(*args, **{ **kwargs, **{ 'worker_id': worker_id } })

def samples_to_outputs(
    *args,
    number_of_samples=1,
    iteration_function=learn.value_iteration,
    num_workers=1,
    convergence_threshold=1e-4
):
    check.check_num_samples(number_of_samples, num_workers)
    check.check_tensor_or_number_args(args)

    output_calculator = func.partial(
        output_sample_calculator_mps,
        number_of_samples=(number_of_samples // num_workers),
        iteration_function=iteration_function,
        convergence_threshold=convergence_threshold,
        value_initializations=None,
        worker_pool_size=num_workers
    )

    with mps.Pool(num_workers) as pool:
        output_samples_list = pool.starmap(
            output_calculator,
            zip(
                range(num_workers),
                *[
                    misc.split_1d_tensor_into_list(
                        tiled_arg, number_of_samples // num_workers
                    ) for tiled_arg in [
                        arg if (
                            hasattr(arg, '__len__') and len(arg) == number_of_samples
                        ) else misc.tile_tensor(arg, number_of_samples) for arg in args
                    ]
                ]
            )
        )
    
    return torch.cat(output_samples_list, axis=0)