from waver.components._utils import location_to_index

def test_location_to_index():
    """Test instantiating a time object."""
    index = location_to_index((10, None, 20), 0.1)

    assert index == (99, slice(None), 199)
