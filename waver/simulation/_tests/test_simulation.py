from waver.simulation import Simulation, run_single_source, run_multiple_sources
import pytest


simulation_variable_params = [
    # 1D full grid, full time
    ({
        'size': (12.8e-3,),
        'location': (0,),
        'temporal_downsample': 1,
        'boundary': 0,
        'edge': None,
    }, {
        'grid_shape': (128,),
        'detected_wave_shape': (400, 128),
    }),
    # 1D full grid, half time
    ({
        'size': (12.8e-3,),
        'location': (0,),
        'temporal_downsample': 2,
        'boundary': 0,
        'edge': None,
    }, {
        'grid_shape': (128,),
        'detected_wave_shape': (200, 128),
    }),
    # 1D full grid, half time, full boundary
    ({
        'size': (12.8e-3,),
        'location': (0,),
        'temporal_downsample': 2,
        'boundary': 1,
        'edge': None,
    }, {
        'grid_shape': (128,),
        'detected_wave_shape': (200, 2),
    }),
    # 1D full grid, half time, thick boundary
    ({
        'size': (12.8e-3,),
        'location': (0,),
        'temporal_downsample': 2,
        'boundary': 5,
        'edge': None,
    }, {
        'grid_shape': (128,),
        'detected_wave_shape': (200, 2 * 5),
    }),
    # 1D full grid, half time, one boundary edge
    ({
        'size': (12.8e-3,),
        'location': (0,),
        'temporal_downsample': 2,
        'boundary': 1,
        'edge': 0,
    }, {
        'grid_shape': (128,),
        'detected_wave_shape': (200, 1),
    }),
    # 2D full grid, full time
    ({
        'size': (12.8e-3, 12.8e-3),
        'location': (0, 0),
        'temporal_downsample': 1,
        'boundary': 0,
        'edge': None,
    }, {
        'grid_shape': (128, 128),
        'detected_wave_shape': (400, 128, 128),
    }),
    # 2D full grid, half time
    ({
        'size': (12.8e-3, 12.8e-3),
        'location': (0, 0),
        'temporal_downsample': 2,
        'boundary': 0,
        'edge': None,
    }, {
        'grid_shape': (128, 128),
        'detected_wave_shape': (200, 128, 128),
    }),
    # 2D full grid, half time, full boundary
    ({
        'size': (12.8e-3, 12.8e-3),
        'location': (0, 0),
        'temporal_downsample': 2,
        'boundary': 1,
        'edge': None,
    }, {
        'grid_shape': (128, 128),
        'detected_wave_shape': (200, 4, 128),
    }),
    # 2D full grid, half time, thick boundary
    ({
        'size': (12.8e-3, 12.8e-3),
        'location': (0, 0),
        'temporal_downsample': 2,
        'boundary': 5,
        'edge': None,
    }, {
        'grid_shape': (128, 128),
        'detected_wave_shape': (200, 4 * 5, 128),
    }),
    # 2D full grid, half time, one boundary edge
    ({
        'size': (12.8e-3, 12.8e-3),
        'location': (0, 0),
        'temporal_downsample': 2,
        'boundary': 1,
        'edge': 0,
    }, {
        'grid_shape': (128, 128),
        'detected_wave_shape': (200, 1, 128),
    }),
   # 3D full grid, full time
    ({
        'size': (3.2e-3, 3.2e-3, 3.2e-3),
        'location': (0, 0, 0),
        'temporal_downsample': 1,
        'boundary': 0,
        'edge': None,
        'pml_thickness': 2,
    }, {
        'grid_shape': (32, 32, 32),
        'detected_wave_shape': (400, 32, 32, 32),
    }),
    # 3D full grid, half time
    ({
        'size': (3.2e-3, 3.2e-3, 3.2e-3),
        'location': (0, 0, 0),
        'temporal_downsample': 2,
        'boundary': 0,
        'edge': None,
        'pml_thickness': 2,
    }, {
        'grid_shape': (32, 32, 32),
        'detected_wave_shape': (200, 32, 32, 32),
    }),
    # 3D full grid, half time, full boundary
    ({
        'size': (3.2e-3, 3.2e-3, 3.2e-3),
        'location': (0, 0, 0),
        'temporal_downsample': 2,
        'boundary': 1,
        'edge': None,
        'pml_thickness': 2,
    }, {
        'grid_shape': (32, 32, 32),
        'detected_wave_shape': (200, 6, 32, 32),
    }),
    # 3D full grid, half time, thick boundary
    ({
        'size': (3.2e-3, 3.2e-3, 3.2e-3),
        'location': (0, 0, 0),
        'temporal_downsample': 2,
        'boundary': 5,
        'edge': None,
        'pml_thickness': 2,
    }, {
        'grid_shape': (32, 32, 32),
        'detected_wave_shape': (200, 6 * 5, 32, 32),
    }),
    # 3D full grid, half time, one boundary edge
    ({
        'size': (3.2e-3, 3.2e-3, 3.2e-3),
        'location': (0, 0, 0),
        'temporal_downsample': 2,
        'boundary': 1,
        'edge': 0,
        'pml_thickness': 2,
    }, {
        'grid_shape': (32, 32, 32),
        'detected_wave_shape': (200, 1, 32, 32),
    }),
]

# Add constant parameters
simulation_params = []
for sim_dict, expected_dict in simulation_variable_params:
    sim_dict.update({
        'spacing': 100e-6,
        'max_speed': 686,
        'time_step': 50e-9,
        'period': 5e-6,
        'duration': 20e-6,        
    })
    simulation_params.append((sim_dict, expected_dict))


@pytest.mark.parametrize("sim_dict, expected_dict", simulation_params)
def test_simulation(sim_dict, expected_dict):
    """Test instantiating and running a simulation."""
    
    # Create simulation
    sim = Simulation(size=sim_dict['size'], spacing=sim_dict['spacing'],
        max_speed=sim_dict['max_speed'], time_step=sim_dict['time_step'],
        pml_thickness=sim_dict.get('pml_thickness', 20))

    # Add source
    sim.add_source(location=sim_dict['location'], period=sim_dict['period'])
    
    # Add detector
    sim.add_detector(boundary=sim_dict['boundary'], edge=sim_dict['edge'])
    
    # Run simulation
    sim.run(duration=sim_dict['duration'], temporal_downsample=sim_dict['temporal_downsample'])

    # Confirm simulation parameters as as expected
    assert sim.time.step == sim_dict['time_step']
    assert sim.grid.shape == expected_dict['grid_shape']

    # Confirm output shapes are as expected
    assert sim.grid_speed.shape == expected_dict['grid_shape']
    assert sim.detected_wave.shape == expected_dict['detected_wave_shape']

    # Note that the dimensionality of the detected wave is always one plus the grid
    assert sim.detected_wave.ndim == sim.grid_speed.ndim + 1


@pytest.mark.parametrize("sim_dict, expected_dict", simulation_params)
def test_single_source(sim_dict, expected_dict):
    """Test running a simlution with a single source."""
    
    # Run simulation with a single source
    detected_wave, grid_speed = run_single_source(**sim_dict)

    # Confirm output shapes are as expected
    assert grid_speed.shape == (1,) + expected_dict['grid_shape']
    assert detected_wave.shape == expected_dict['detected_wave_shape']

    # Note that the dimensionality of the detecteds wave matches grid
    assert detected_wave.ndim == grid_speed.ndim


@pytest.mark.parametrize("n_sources", [1, 2])
@pytest.mark.parametrize("sim_dict, expected_dict", simulation_params)
def test_multiple_source(sim_dict, expected_dict, n_sources):
    """Test running a simlution with a multiple sources."""

    # Add sources
    sim_dict = sim_dict.copy()
    sources = n_sources * [{'location': sim_dict['location'], 'period': sim_dict['period']}]
    del sim_dict['location']
    del sim_dict['period']
    sim_dict['sources'] = sources

    # Run simulation with multiple sources
    detected_waves, grid_speed = run_multiple_sources(**sim_dict)

    # Confirm output shapes are as expected
    assert grid_speed.shape == (1, 1) + expected_dict['grid_shape']
    assert detected_waves.shape == (n_sources,) + expected_dict['detected_wave_shape']

    # Note that the dimensionality of the detecteds wave matches grid
    assert detected_waves.ndim == grid_speed.ndim
