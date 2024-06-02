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
# normalize_z_score_folder('/newdata3/chf/test_data/old_no_normalized/snr100/5LQW',
#                          '/newdata3/chf/test_data/snr100/5LQW')
#
# normalize_z_score_folder('/newdata3/chf/test_data/old_no_normalized/snr01/5LQW',
#                          '/newdata3/chf/test_data/snr01/5LQW')
#
# normalize_z_score_folder('/newdata3/chf/test_data/old_no_normalized/snr005/5LQW',
#                          '/newdata3/chf/test_data/snr005/5LQW')
#
# normalize_z_score_folder('/newdata3/chf/test_data/old_no_normalized/snr003/5LQW',
#                          '/newdata3/chf/test_data/snr003/5LQW')
#
# normalize_z_score_folder('/newdata3/chf/test_data/old_no_normalized/snr001/5LQW',
#                          '/newdata3/chf/test_data/snr001/5LQW')
#
#
#
#
#
# normalize_z_score_folder('/newdata3/chf/train_data/old_no_normalized/snr100/5LQW',
#                          '/newdata3/chf/train_data/snr100/5LQW')
#
# normalize_z_score_folder('/newdata3/chf/train_data/old_no_normalized/snr100/5LQW',
#                          '/newdata3/chf/train_data/snr01/5LQW')
#
# normalize_z_score_folder('/newdata3/chf/train_data/old_no_normalized/snr100/5LQW',
#                          '/newdata3/chf/train_data/snr005/5LQW')
#
# normalize_z_score_folder('/newdata3/chf/train_data/old_no_normalized/snr100/5LQW',
#                          '/newdata3/chf/train_data/snr003/5LQW')
#
# normalize_z_score_folder('/newdata3/chf/train_data/old_no_normalized/snr100/5LQW',
#                          '/newdata3/chf/train_data/snr001/5LQW')


# normalize_z_score_folder('/newdata3/chf/train_data_wedge/snr001/5LQW_no_normalized',
#                          '/newdata3/chf/train_data_wedge/snr001/5LQW')
#
# normalize_z_score_folder('/newdata3/chf/test_data_wedge/snr001/5LQW_no_normalized',
#                          '/newdata3/chf/test_data_wedge/snr001/5LQW')


# normalize_z_score_folder('/newdata3/chf/train_data_wedge/snr001/5MPA_no_normalized',
#                          '/newdata3/chf/train_data_wedge/snr001/5MPA')
# normalize_z_score_folder('/newdata3/chf/test_data_wedge/snr001/5MPA_no_normalized',
#                          '/newdata3/chf/test_data_wedge/snr001/5MPA')
# normalize_z_score_folder('/newdata3/chf/train_data_wedge/snr001/5T2C_no_normalized',
#                          '/newdata3/chf/train_data_wedge/snr001/5T2C')
# normalize_z_score_folder('/newdata3/chf/test_data_wedge/snr001/5T2C_no_normalized',
#                          '/newdata3/chf/test_data_wedge/snr001/5T2C')
# normalize_z_score_folder('/newdata3/chf/train_data_wedge/snr001/6A5L_no_normalized',
#                          '/newdata3/chf/train_data_wedge/snr001/6A5L')
# normalize_z_score_folder('/newdata3/chf/test_data_wedge/snr001/6A5L_no_normalized',
#                          '/newdata3/chf/test_data_wedge/snr001/6A5L')

"5LQW, 5MPA, 5T2C, 6A5L"
"snr100, snr01, snr005, snr003, snr001"
# normalize_z_score_folder("/newdata3/chf/normalized/wedge/no_normalized/snr100/5LQW",
#                          "/newdata3/chf/normalized/wedge/snr100/5LQW")
# normalize_z_score_folder("/newdata3/chf/normalized/wedge/no_normalized/snr100/5MPA",
#                          "/newdata3/chf/normalized/wedge/snr100/5MPA")
# normalize_z_score_folder("/newdata3/chf/normalized/wedge/no_normalized/snr100/5T2C",
#                          "/newdata3/chf/normalized/wedge/snr100/5T2C")
# normalize_z_score_folder("/newdata3/chf/normalized/wedge/no_normalized/snr100/6A5L",
#                          "/newdata3/chf/normalized/wedge/snr100/6A5L")
#
# normalize_z_score_folder("/newdata3/chf/normalized/wedge/no_normalized/snr01/5LQW",
#                          "/newdata3/chf/normalized/wedge/snr01/5LQW")
# normalize_z_score_folder("/newdata3/chf/normalized/wedge/no_normalized/snr01/5MPA",
#                          "/newdata3/chf/normalized/wedge/snr01/5MPA")
# normalize_z_score_folder("/newdata3/chf/normalized/wedge/no_normalized/snr01/5T2C",
#                          "/newdata3/chf/normalized/wedge/snr01/5T2C")
# normalize_z_score_folder("/newdata3/chf/normalized/wedge/no_normalized/snr01/6A5L",
#                          "/newdata3/chf/normalized/wedge/snr01/6A5L")
#
# normalize_z_score_folder("/newdata3/chf/normalized/wedge/no_normalized/snr005/5LQW",
#                          "/newdata3/chf/normalized/wedge/snr005/5LQW")
# normalize_z_score_folder("/newdata3/chf/normalized/wedge/no_normalized/snr005/5MPA",
#                          "/newdata3/chf/normalized/wedge/snr005/5MPA")
# normalize_z_score_folder("/newdata3/chf/normalized/wedge/no_normalized/snr005/5T2C",
#                          "/newdata3/chf/normalized/wedge/snr005/5T2C")
# normalize_z_score_folder("/newdata3/chf/normalized/wedge/no_normalized/snr005/6A5L",
#                          "/newdata3/chf/normalized/wedge/snr005/6A5L")
#
# normalize_z_score_folder("/newdata3/chf/normalized/wedge/no_normalized/snr003/5LQW",
#                          "/newdata3/chf/normalized/wedge/snr003/5LQW")
# normalize_z_score_folder("/newdata3/chf/normalized/wedge/no_normalized/snr003/5MPA",
#                          "/newdata3/chf/normalized/wedge/snr003/5MPA")
# normalize_z_score_folder("/newdata3/chf/normalized/wedge/no_normalized/snr003/5T2C",
#                          "/newdata3/chf/normalized/wedge/snr003/5T2C")
# normalize_z_score_folder("/newdata3/chf/normalized/wedge/no_normalized/snr003/6A5L",
#                          "/newdata3/chf/normalized/wedge/snr003/6A5L")