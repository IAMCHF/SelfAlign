import mrcfile
import numpy as np
from SelfAlign.preprocessing.img_processing import generate_mask

input_data_shape = (40, 40, 40)
mask = np.zeros(input_data_shape, dtype=np.float32)

center = np.array([20.0, 20.0, 20.0])
radius = 20.0

for i in range(input_data_shape[0]):
    for j in range(input_data_shape[1]):
        for k in range(input_data_shape[2]):
            if np.linalg.norm(np.array([i, j, k]) - center) <= radius:
                mask[i, j, k] = 1.0
with mrcfile.new('mask_binned5.mrc', overwrite=True) as mrc:
    mrc.set_data(mask)

mask_wedge = generate_mask(input_data_shape)
with mrcfile.new('mask_wedge_binned5.mrc', overwrite=True) as mrc:
    mrc.set_data(mask_wedge)

# mask = np.expand_dims(mask, axis=0).repeat(batch_size, axis=0)
# masked_data = input_data * mask
# output = conv_layer(masked_data)