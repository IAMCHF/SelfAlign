import mrcfile
import numpy as np
from scipy.stats import zscore


def normalize_z_score_single_file(input_filepath, output_filepath, mask_filepath="mask_32.mrc"):
    # Load the mask if needed
    with mrcfile.open(mask_filepath, permissive=True) as mask_mrc:
        mask_binned5 = mask_mrc.data.astype(np.float32)

    # Open the input MRC file
    with mrcfile.open(input_filepath, permissive=True) as mrc_input:
        mrc_data = mrc_input.data.astype(np.float32)

    # Apply the mask
    masked_data = mrc_data * mask_binned5

    # Flatten the masked data for Z-score normalization
    flat_data = masked_data.flatten()

    # Perform Z-score normalization
    normalized_data = zscore(flat_data, axis=None)

    # Reshape the normalized data back to its original shape
    reshaped_data = normalized_data.reshape(masked_data.shape)

    # Save the normalized data to a new MRC file
    with mrcfile.new(output_filepath, overwrite=True) as mrc_output:
        mrc_output.set_data(reshaped_data.astype(np.float32))

    print(f"Finished Z-score normalization. The result is saved to {output_filepath}")

