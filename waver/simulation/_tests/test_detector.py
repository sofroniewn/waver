import numpy as np
import pytest
from waver.simulation._detector import Detector


params = [
    # 1D full grid
    ({
        'shape': (128,),
        'spacing': (1,),
        'boundary': 0,
        'edge': None,
    }, {
        'downsample_shape': (128,),
    }),
    # 1D full boundary
    ({
        'shape': (128,),
        'spacing': (1,),
        'boundary': 1,
        'edge': None,
    }, {
        'downsample_shape': (2,),
    }),
    # 1D thick boundary
    ({
        'shape': (128,),
        'spacing': (1,),
        'boundary': 10,
        'edge': None,
    }, {
        'downsample_shape': (2 * 10,),
    }),
    # 1D single edge
    ({
        'shape': (128,),
        'spacing': (1,),
        'boundary': 1,
        'edge': 1,
    }, {
        'downsample_shape': (1,),
    }),
    # 2D full grid
    ({
        'shape': (128, 128),
        'spacing': (1,),
        'boundary': 0,
        'edge': None,
    }, {
        'downsample_shape': (128, 128),
    }),
    # 2D full boundary
    ({
        'shape': (128, 128),
        'spacing': (1,),
        'boundary': 1,
        'edge': None,
    }, {
        'downsample_shape': (4, 128),
    }),
    # 2D thick boundary
    ({
        'shape': (128, 128),
        'spacing': (1,),
        'boundary': 10,
        'edge': None,
    }, {
        'downsample_shape': (4 * 10, 128),
    }),
    # 2D single edge
    ({
        'shape': (128, 128),
        'spacing': (1,),
        'boundary': 1,
        'edge': 1,
    }, {
        'downsample_shape': (1, 128),
    }),
    # 3D full grid
    ({
        'shape': (128, 128, 128),
        'spacing': (1,),
        'boundary': 0,
        'edge': None,
    }, {
        'downsample_shape': (128, 128, 128),
    }),
    # 3D full boundary
    ({
        'shape': (128, 128, 128),
        'spacing': (1,),
        'boundary': 1,
        'edge': None,
    }, {
        'downsample_shape': (6, 128, 128),
    }),
    # 3D thick boundary
    ({
        'shape': (128, 128, 128),
        'spacing': (1,),
        'boundary': 10,
        'edge': None,
    }, {
        'downsample_shape': (6 * 10, 128, 128),
    }),
    # 3D single edge
    ({
        'shape': (128, 128, 128),
        'spacing': (1,),
        'boundary': 1,
        'edge': 1,
    }, {
        'downsample_shape': (1, 128, 128),
    }),
]


@pytest.mark.parametrize("detector_params, expected_params", params)
def test_detector(detector_params, expected_params):
    """Test instantiating a detector."""
    detector = Detector(**detector_params)

    assert detector.shape == detector_params['shape']
    assert len(detector.downsample_shape) == len(detector.shape)
    assert detector.downsample_shape == expected_params['downsample_shape']

    # Record wave
    wave = np.zeros(detector_params['shape'])
    detected_wave = detector.sample(wave)
    assert detected_wave.shape == expected_params['downsample_shape']

    # Note that sampling never changes the dimensionality of the wave
    assert wave.ndim == detected_wave.ndim
