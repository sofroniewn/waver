from waver.simulation._utils import fourier_sample
import napari


shape = (32, 32)
values = fourier_sample(shape)

viewer = napari.Viewer()
viewer.add_image(values, colormap='viridis', contrast_limits=[0, 1], name='fourier sample',
    interpolation='bilinear')

napari.run()