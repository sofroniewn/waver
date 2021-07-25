from waver.simulation._utils import location_to_index

def test_location_to_index():
    """Test instantiating a time object."""
    index = location_to_index((10, None, 20), 0.1, (100,))

    assert index == (99,)
