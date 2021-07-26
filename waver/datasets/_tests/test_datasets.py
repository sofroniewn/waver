from tempfile import TemporaryDirectory
from waver.datasets import generate_simulation_dataset, load_simulation_dataset


def test_dataset_generator_and_loader():
    """Test generating and loading a dateset."""
    runs = 4
    sim_params = {
        'size': (3.2e-3, 3.2e-3),
        'spacing': 100e-6,
        'duration': 20e-6,
        'min_speed': 343,
        'max_speed': 686,
        'speed': 686,
        'time_step': 50e-9,
        'temporal_downsample': 2,
        'sources': [{
            'location': (1.6e-3, 1.6e-3),
            'period': 5e-6,
            'ncycles':1,
        }],
        'boundary': 1,
        'edge': 1
    }
    with TemporaryDirectory(suffix='.zarr') as path:
        generate_simulation_dataset(path, runs, **sim_params)
        dataset = load_simulation_dataset(path)

        assert len(dataset) == 2
        assert dataset[0][0].shape == (4, 1, 200, 1, 32)
        assert dataset[1][0].shape == (4, 1, 1, 32, 32)