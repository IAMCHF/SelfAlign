import numpy as np
import mrcfile
# from scipy.fftpack import fftn, ifftn
from SelfAlign.preprocessing.simulate import mw2d

with mrcfile.open("mask_32.mrc",
                  permissive=True) as mrcData:
    mask_binned5 = mrcData.data.astype(np.float32)


def get_mrc_data(mrc_path):
    with mrcfile.open(mrc_path, permissive=True) as cdata:
        cdata = cdata.data.astype(np.float32)
    return cdata * mask_binned5


def read_parameters(filename):
    with open(filename, 'r') as params_file:
        lines = params_file.readlines()
    parameters_list, x1, x2, x3, x4 = [], [], [], [], []
    for line in lines:
        parts = line.strip().split("\t")
        ori_path, temple_path = parts[0], parts[1]
        # quaternion = np.array([float(q) for q in parts[2:6]])
        # translation = np.array([float(t) for t in parts[6:]])
        # parameters_list.append((ori_path, temple_path, quaternion, translation))
        euler_angles = np.array([float(q) for q in parts[2:5]])
        translation = np.array([float(t) for t in parts[5:]])
        x1.append(ori_path)
        x2.append(temple_path)
        x3.append(euler_angles)
        x4.append(translation)
    parameters_list.append(x1)
    parameters_list.append(x2)
    parameters_list.append(x3)
    parameters_list.append(x4)
    return parameters_list


def generate_mask(shape):
    dim_z, dim_x, dim_y = shape
    two_d_mask = mw2d(dim_x)
    mask = np.broadcast_to(two_d_mask[:, np.newaxis, :], (dim_z, dim_x, dim_y))
    return mask.astype(np.float32)
#
#
# def low_pass_gaussian_filter(image, sigma):
#     f = fftn(image.astype(np.complex64))
#     freq_vectors = [np.fft.fftfreq(dim_size) * dim_size for dim_size in image.shape]
#     frequencies = np.stack(np.meshgrid(*freq_vectors, indexing='ij'), axis=-1)
#     frequencies_norm = np.linalg.norm(frequencies, axis=-1)
#     frequency_domain_filter = np.exp(-(frequencies_norm ** 2 / (2 * sigma ** 2))) / (sigma * np.sqrt(2 * np.pi))
#     f *= frequency_domain_filter
#     filtered_image = np.real(ifftn(f))
#     return filtered_image
