import torch
import functools as func
import operator as op

################################################################################

TINY_NUMBER = 1e-45 # To avoid both divide by 0 and overflow in torch

################################################################################

def build_sweep_config(
    sweep_name=None,
    sweep_id=None,
    sweep_params=None,
    sweep_description=None,
    project=None,
    entity=None
):
    return {
        'parameters': { # Allows us to name individual runs with preset strings
            key: (value_dict if ('value' in sweep_params.get(key)) else { 'values': [
                [val, name] for val, name in zip(value_dict.get('values'), value_dict.get('names'))
            ]}) for (key, value_dict) in sweep_params.items()
        },
        'name': '{0}-{1}'.format(sweep_id, sweep_name),
        'description': sweep_description,
        'project': project,
        'entity': entity,
        'method': 'grid'
    }

def retrieve(dictionary, path_with_dots):
    return func.reduce(op.getitem, path_with_dots.split('.'), dictionary)

# Biggest fractional change in state value over one iteration.
def calculate_value_convergence(old_values, new_values, tiny_number=TINY_NUMBER):
    return torch.max(torch.abs((old_values - new_values) / (old_values + tiny_number)))

def pdf_sampler_constructor(pdf=lambda x: 1, interval=(0, 1), resolution=100):

# A bit weird, but lets us match the currying signature of PyTorch distribution functions.
    def constructor_l2():

        def pdf_sampler(number_of_samples):
            source_samples = torch.rand(number_of_samples)

            dest_boundary_points = torch.linspace(*(interval + (resolution,)))
            sample_widths = dest_boundary_points[1:] - dest_boundary_points[:-1]
            pdf_at_midpoints = torch.ones(resolution - 1) * pdf(
                0.5 * (dest_boundary_points[1:] + dest_boundary_points[:-1])
            )
            probability_masses = sample_widths * pdf_at_midpoints / torch.sum(sample_widths * pdf_at_midpoints)

            source_boundary_points = torch.cat((torch.zeros(1), torch.cumsum(probability_masses, 0)))
            sample_bins = torch.bucketize(source_samples, source_boundary_points)
            
            sample_bin_positions = (
                source_samples - source_boundary_points[sample_bins - 1]
            ) / (
                source_boundary_points[sample_bins] - source_boundary_points[sample_bins - 1]
            )

            return sample_bin_positions * (
                dest_boundary_points[sample_bins] - dest_boundary_points[sample_bins - 1]
            ) + dest_boundary_points[sample_bins - 1]
        
        return pdf_sampler
    
    return constructor_l2