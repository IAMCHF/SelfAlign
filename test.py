import numpy as np
import os
import mrcfile
import multiprocessing as mp
from scipy.stats import zscore


def binning_3d(file_path, output_dir, bin_size):
    with mrcfile.open(file_path) as mrc:
        data = mrc.data.astype(np.float32)

    z_dim, y_dim, x_dim = data.shape
    assert z_dim % bin_size == 0 and y_dim % bin_size == 0 and x_dim % bin_size == 0
    downsampled_data = data.reshape((z_dim // bin_size, bin_size,
                                     y_dim // bin_size, bin_size,
                                     x_dim // bin_size, bin_size)).mean(axis=(1, 3, 5))
    filename = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = filename + '_binned' + str(bin_size) + '.mrc'
    output_file = os.path.join(output_dir, output_filename)
    with mrcfile.new(output_file, overwrite=False) as mrc:
        mrc.set_data(downsampled_data)



binning_3d('/media/hao/Hard_disk_1T/datasets/10045/1/EMD-3228.mrc',
               '/media/hao/Hard_disk_1T/datasets/10045/1',
               bin_size=5)
#
# output_file = "/media/hao/Sata500g/my_dataset/class_binned4/run2_class001_binned4_normalized.mrc"
# with mrcfile.new(output_file, overwrite=True) as mrc:
#     mrc.set_data(normalize_z_score_data(data))
# mrc_files = [os.path.join('/media/hao/Hard_disk_1T/datasets/10045/AnticipatedResults/Particles/Tomograms/subtomos_good/', f)
#              for f in
#              os.listdir('/media/hao/Hard_disk_1T/datasets/10045/AnticipatedResults/Particles/Tomograms/subtomos_good/')
#              if f.endswith('.mrc')]
# ouput_dir = '/media/hao/Hard_disk_1T/datasets/10045/AnticipatedResults/Particles/Tomograms/subtomos_bin5'
#
# with mp.Pool(processes=22) as pool:
#     pool.starmap(binning_3d, [(file_path, ouput_dir, 5) for file_path in mrc_files])


# with mrcfile.open("/media/hao/Hard_disk_1T/datasets/my_dataset/run2_class001.mrc") as mrc:
#     data = mrc.data.astype(np.float32)
# with mrcfile.open("mask_binned4.mrc",
#                   permissive=True) as mrcData:
#     mask_binned4 = mrcData.data.astype(np.float32)
# def normalize_z_score(mrc_path):
#     with mrcfile.open(mrc_path, permissive=True) as mrcData:
#         mrc_data = mrcData.data.astype(np.float32)
#     normalized_data = zscore(mrc_data, axis=None)
#     normalized_data = normalized_data.reshape(mrc_data.shape)
#     return normalized_data
    # return normalized_data * mask_binned4


#
#
# def normalize_z_score_data(mrc_data):
#     normalized_data = zscore(mrc_data, axis=None)
#     normalized_data = normalized_data.reshape(mrc_data.shape)
#     return normalized_data
#
#
# normalize_z_data = normalize_z_score(
#     '/media/hao/Hard_disk_1T/datasets/my_dataset/class_binned5/run2_class001_binned5.mrc')
# with mrcfile.new('/media/hao/Hard_disk_1T/datasets/my_dataset/class_binned5/run2_class001_binned5_normalized.mrc',
#                  overwrite=True) as mrc:
#     mrc.set_data(normalize_z_data)
