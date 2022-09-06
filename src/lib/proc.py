import sys
import torch
import functools as func
import multiprocessing as mps

from .utils import learn
from .utils import misc
from . import check

def is_manual_sparse_tensor(input_to_check):
    return hasattr(input_to_check, 'keys') and set(input_to_check.keys()) == set(['indices', 'values', 'size'])

# Multiprocessing doesn't support sparse tensors, see
# https://github.com/pytorch/pytorch/issues/20248#issuecomment-490525946
# So what we do instead is create a dict with the sparse tensor indices, values, and shape; and
# then reconstruct the sparse tensor from that inside each worker. This is only needed for
# multiprocessing.
def sparse_tensor_to_manual_sparse_tensor(sparse_tensor):
    coalesced_tensor = sparse_tensor.coalesce()
    return {
        'indices': coalesced_tensor.indices(),
        'values': coalesced_tensor.values(),
        'size': coalesced_tensor.size()
    }

def manual_sparse_tensor_to_sparse_tensor(manual_sparse_tensor):
    return torch.sparse_coo_tensor(
        manual_sparse_tensor['indices'],
        manual_sparse_tensor['values'],
        tuple(manual_sparse_tensor['size'])
    )

def split_1d_tensor_into_list_for_multiprocess(input_object, chunk_size):
    input_tensor = input_object if torch.is_tensor(input_object) else torch.tensor(input_object)

    if input_tensor.is_sparse:
        coalesced_tensor = input_tensor.coalesce()
        number_of_chunks = coalesced_tensor.size()[0] // chunk_size
        total_indices_per_chunk = coalesced_tensor.indices().shape[1] // number_of_chunks
        selection_indices = coalesced_tensor.indices()[0][:total_indices_per_chunk]

        return [
            sparse_tensor_to_manual_sparse_tensor(
                torch.sparse_coo_tensor(
                    torch.cat((selection_indices.unsqueeze(0), indices), dim=0),
                    values,
                    (chunk_size,) + tuple(coalesced_tensor.size()[1:])
                ).coalesce()
            ) for indices, values in zip(
                torch.split(coalesced_tensor.indices()[1:], total_indices_per_chunk, dim=1),
                torch.split(coalesced_tensor.values(), total_indices_per_chunk, dim=0)
            )
        ]

    else:
        return [tensor for tensor in torch.split(input_tensor, chunk_size, dim=0)]

def is_input_already_tiled(input_object, number_of_samples):
    if torch.is_tensor(input_object):
        return input_object.size()[0] == number_of_samples
    else:
        return hasattr(input_object, '__len__') and len(input_object) == number_of_samples

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
    # Multiprocessing can't handle sparse tensors, so this hack transports the parameters
    # in using a manually defined data structure
    input_args = [
        manual_sparse_tensor_to_sparse_tensor(arg_tensor) if (
            is_manual_sparse_tensor(arg_tensor)
        ) else arg_tensor for arg_tensor in args
    ]
    return output_sample_calculator(*input_args, **{ **kwargs, **{ 'worker_id': worker_id } })

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
                    split_1d_tensor_into_list_for_multiprocess(
                        tiled_arg, number_of_samples // num_workers
                    ) for tiled_arg in [
                        arg if (
                            is_input_already_tiled(arg, number_of_samples)
                        ) else misc.tile_tensor(arg, number_of_samples) for arg in args
                    ]
                ]
            )
        )
    
    return torch.cat(output_samples_list, axis=0)