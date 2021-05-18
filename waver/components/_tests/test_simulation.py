from waver.components.simulation import Simulation

def test_simulation():
    """Test instantiating a simulation."""
    sim = Simulation(size=(0.1, 0.1), spacing=1e-2, speed=343, duration=1e-2)

    assert sim.grid.ndim == 2
