from waver.inversion._dataset import WaverSimulationDataset

# Define root path for simulation
path = '/Users/nsofroniew/Documents/inverting_physics/tests_003/'

ds = WaverSimulationDataset(path)

print('Dataset : ', ds)
print('Input, Output shapes : ', ds[0][0].shape, ds[0][1].shape)