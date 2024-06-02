import os
import glob
import mrcfile
import numpy as np
from scipy.stats import zscore

with mrcfile.open("mask_32.mrc",
                  permissive=True) as mrcData:
    mask_binned5 = mrcData.data.astype(np.float32)


def normalize_z_score_folder(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Gather all MRC files in the input folder
    mrc_files = glob.glob(os.path.join(input_folder, '*.mrc'))

    # Initialize an array to hold all MRC data
    all_mrc_data = []

    for mrc_path in mrc_files:
        with mrcfile.open(mrc_path, permissive=True) as mrc_file:
            mrc_data = mrc_file.data.astype(np.float32)

        # Apply the mask and append the masked data to the list
        masked_data = mrc_data * mask_binned5
        all_mrc_data.append(masked_data.flatten())

    # Concatenate all masked data into a single array for global normalization
    concatenated_data = np.concatenate(all_mrc_data, axis=0)

    # Perform global Z-score normalization
    globally_normalized_data = zscore(concatenated_data, axis=None)

    # Split the globally normalized data back into individual subtomograms
    num_subtomograms = len(all_mrc_data)
    split_normalized_data = np.split(globally_normalized_data, num_subtomograms)

    original_shape = mask_binned5.shape
    # Reshape and save each normalized subtomogram to the output folder
    for i, (mrc_path, normalized_subtomogram) in enumerate(zip(mrc_files, all_mrc_data)):
        reshaped_subtomogram = normalized_subtomogram.reshape(original_shape)
        output_filename = os.path.basename(mrc_path)
        output_path = os.path.join(output_folder, output_filename)
        with mrcfile.new(output_path) as mrc_output:
            mrc_output.set_data(reshaped_subtomogram.astype(np.float32))
    print(f"Finished global Z-score normalization. Results saved to {output_folder}")


# Call the function with your input and output folders and the mask
"5LQW, 5MPA, 5T2C, 6A5L"
"snr100, snr01, snr005, snr003, snr001"
"train_data_wedge, test_data_wedge, train_data_rotation_strategy, test_data_rotation_strategy"

rota1 = "/newdata3/chf/test_data_rotation_strategy/snr100/5T2C"
os.rename(rota1, rota1 + "_no_wedge")
rota2 = "/newdata3/chf/test_data_rotation_strategy/snr01/5T2C"
os.rename(rota2, rota2 + "_no_wedge")
rota3 = "/newdata3/chf/test_data_rotation_strategy/snr005/5T2C"
os.rename(rota3, rota3 + "_no_wedge")
rota4 = "/newdata3/chf/test_data_rotation_strategy/snr003/5T2C"
os.rename(rota4, rota4 + "_no_wedge")
rota5 = "/newdata3/chf/test_data_rotation_strategy/snr001/5T2C"
os.rename(rota5, rota5 + "_no_wedge")

rota6 = "/newdata3/chf/train_data_rotation_strategy/snr100/5T2C"
os.rename(rota6, rota6 + "_no_wedge")
rota7 = "/newdata3/chf/train_data_rotation_strategy/snr01/5T2C"
os.rename(rota7, rota7 + "_no_wedge")
rota8 = "/newdata3/chf/train_data_rotation_strategy/snr005/5T2C"
os.rename(rota8, rota8 + "_no_wedge")
rota9 = "/newdata3/chf/train_data_rotation_strategy/snr003/5T2C"
os.rename(rota9, rota9 + "_no_wedge")
rota10 = "/newdata3/chf/train_data_rotation_strategy/snr001/5T2C"
os.rename(rota10, rota10 + "_no_wedge")

normalize_z_score_folder(rota1 + "_no_normalized", rota1)
normalize_z_score_folder(rota2 + "_no_normalized", rota2)
normalize_z_score_folder(rota3 + "_no_normalized", rota3)
normalize_z_score_folder(rota4 + "_no_normalized", rota4)
normalize_z_score_folder(rota5 + "_no_normalized", rota5)
normalize_z_score_folder(rota6 + "_no_normalized", rota6)
normalize_z_score_folder(rota7 + "_no_normalized", rota7)
normalize_z_score_folder(rota8 + "_no_normalized", rota8)
normalize_z_score_folder(rota9 + "_no_normalized", rota9)
normalize_z_score_folder(rota10 + "_no_normalized", rota10)