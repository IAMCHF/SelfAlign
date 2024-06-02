import logging
import os
import random
import time

import mrcfile
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
from multiprocessing import Pool, Manager
import numpy as np
from functools import partial

from SelfAlign.util.metadata import MetaData
from normalize_z_score_foler import normalize_z_score_folder


def save_params_to_txt(outfile, data_list):
    with open(outfile, 'a') as params_file:
        for item in data_list:
            line_to_write = f"{item['source_path']}\t{item['temple_path']}\t"
            transformation_str = ["{:.16f}".format(ti) for ti in item['combined_params']]
            line_to_write += "\t".join(transformation_str)
            line_to_write += "\n"
            params_file.write(line_to_write)


# theta_range = np.linspace(np.pi / 1000, np.pi, num=180, endpoint=False)
# phi_range = np.linspace(np.pi / 1000, 2 * np.pi, num=360, endpoint=False)

theta_range = np.linspace(np.pi / 1000, np.pi / 6.0, num=90, endpoint=False)
phi_range = np.linspace(np.pi / 1000, 2 * np.pi / 6.0, num=180, endpoint=False)

manager = Manager()
temp_data_list = manager.list()


#
# def get_cubes(inp, rota_folder):
#     mrc, start = inp
#     with mrcfile.open(mrc) as mrcData:
#         orig_data = mrcData.data.astype(np.float32)
#     center = (np.array(orig_data.shape) - 1) / 2.
#     batch_len = 100
#     batch_temple_mrcs = []
#     batch_indices = []
#     np.random.seed(42)
#     for _ in range(20000):
#         theta_z1 = np.random.uniform(low=-np.pi, high=np.pi)
#         theta_y = np.random.uniform(low=-np.pi, high=np.pi)
#         theta_z2 = np.random.uniform(low=-np.pi, high=np.pi)
#         euler_angles = [theta_z1, theta_y, theta_z2]
#         rot_matrix = Rotation.from_euler('zyz', euler_angles, degrees=False).as_matrix()
#         rotation_obj = Rotation.from_matrix(rot_matrix)
#         offset = center - np.dot(rot_matrix, center)
#         rotated = affine_transform(orig_data, rot_matrix, offset=offset)
#
#         translation = np.random.uniform(-4.0, 4.0, size=(3,))
#         translation_matrix = np.eye(4)
#         translation_matrix[:3, 3] = translation
#         translated = affine_transform(rotated, translation_matrix, order=0, mode='constant')
#
#         inverse_rotation_obj = rotation_obj.inv()
#         theta_z1, theta_y, theta_z2 = inverse_rotation_obj.as_euler('zyz', degrees=False)
#         if not all(-np.pi <= angle <= np.pi for angle in [theta_z1, theta_y, theta_z2]):
#             print(f"**************** euler_angles out of [-pi, pi] ******")
#         euler_angles = [theta_z1, theta_y, theta_z2]
#         combined_params = euler_angles + list(-translation)
#         batch_temple_mrcs.append(translated)
#         batch_indices.append(start)
#         temp_data_list.append({
#             'source_path': '{}/rotated_{}.mrc'.format(rota_folder, start),
#             'temple_path': mrc,
#             'combined_params': combined_params
#         })
#         if len(batch_temple_mrcs) == batch_len:
#             for temple_mrc, start in zip(batch_temple_mrcs, batch_indices):
#                 with mrcfile.new('{}/rotated_{}.mrc'.format(rota_folder, start), overwrite=True) as output_mrc:
#                     output_mrc.set_data(temple_mrc.astype(np.float32))
#             batch_temple_mrcs = []
#             batch_indices = []
#         start += 1
#     if batch_temple_mrcs:
#         for temple_mrc, start in zip(batch_temple_mrcs, batch_indices):
#             with mrcfile.new('{}/rotated_{}.mrc'.format(rota_folder, start), overwrite=True) as output_mrc:
#                 output_mrc.set_data(temple_mrc.astype(np.float32))


def get_cubes(inp, rota_folder):
    mrc, start = inp
    with mrcfile.open(mrc) as mrcData:
        orig_data = mrcData.data.astype(np.float32)
    center = (np.array(orig_data.shape) - 1) / 2.
    batch_len = 1000
    batch_temple_mrcs = []
    batch_indices = []
    np.random.seed(41)
    for theta in theta_range:
        for phi in phi_range:
            w = np.cos(theta / 2) * np.cos(phi / 2)
            x = np.sin(theta / 2) * np.cos(phi / 2)
            y = np.sin(theta / 2) * np.sin(phi / 2)
            z = np.cos(theta / 2) * np.sin(phi / 2)
            quaternion = [w, x, y, z]
            rot_matrix = Rotation.from_quat(quaternion).as_matrix()
            rotation_obj = Rotation.from_matrix(rot_matrix)
            offset = center - np.dot(rot_matrix, center)
            rotated = affine_transform(orig_data, rot_matrix, offset=offset)

            translation = np.random.uniform(-4.0, 4.0, size=(3,))
            translation_matrix = np.eye(4)
            translation_matrix[:3, 3] = translation
            translated = affine_transform(rotated, translation_matrix, order=0, mode='constant')

            inverse_rotation_obj = rotation_obj.inv()
            theta_z1, theta_y, theta_z2 = inverse_rotation_obj.as_euler('zyz', degrees=False)
            if not all(-np.pi <= angle <= np.pi for angle in [theta_z1, theta_y, theta_z2]):
                print(f"**************** euler_angles out of [-pi, pi] ******")
            euler_angles = [theta_z1, theta_y, theta_z2]
            combined_params = euler_angles + list(-translation)
            # combined_params = euler_angles + list(offset-translation)  # test for rotation and translation strategy
            batch_temple_mrcs.append(translated)
            batch_indices.append(start)
            temp_data_list.append({
                'source_path': '{}/rotated_{}.mrc'.format(rota_folder, start),
                'temple_path': mrc,
                'combined_params': combined_params
            })
            if len(batch_temple_mrcs) == batch_len:
                for temple_mrc, start in zip(batch_temple_mrcs, batch_indices):
                    with mrcfile.new('{}/rotated_{}.mrc'.format(rota_folder, start)) as output_mrc:
                        output_mrc.set_data(temple_mrc.astype(np.float32))
                batch_temple_mrcs = []
                batch_indices = []
            start += 1
    if batch_temple_mrcs:
        for temple_mrc, start in zip(batch_temple_mrcs, batch_indices):
            with mrcfile.new('{}/rotated_{}.mrc'.format(rota_folder, start)) as output_mrc:
                output_mrc.set_data(temple_mrc.astype(np.float32))


def get_cubes_list():
    import os
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    inp = []
    md = MetaData()
    md.read(subtomo_star)
    all_mrc_list = []
    for i, it in enumerate(md):
        if "rlnImageName" in md.getLabels():
            all_mrc_list.append(it.rlnImageName)
    for i, mrc in enumerate(all_mrc_list):
        # inp.append((mrc, int(i * 20000)))
        # inp.append((mrc, int(i * 64800)))
        inp.append((mrc, int(i * 16200)))
    if not os.path.exists(rota):
        os.makedirs(rota)
    rota_folder = rota
    for i in inp:
        get_cubes(i, rota_folder)
    train_outfile = '{}/6A5L.txt'.format(data_dir)
    train_data_list = [d for i, d in enumerate(temp_data_list)]
    save_params_to_txt(train_outfile, train_data_list)
    del temp_data_list[:]


"5LQW, 5MPA, 5T2C, 6A5L"
"snr100, snr01, snr005, snr003, snr001"
"train_data, train_data_067, train_data_133, train_data_2, train_data_random, train_data_rotation_strategy"

data_dir = "/newdata3/chf/train_data_rotation_strategy/snr001"
rota = data_dir + "/6A5L"
subtomo_star = "/HBV/Caohaofan/selfalign/6A5L.star"

get_cubes_list()
normalize_z_score_folder(rota, rota + "_1")
os.rename(rota, rota + "_no_normalized")
os.rename(rota + "_1", rota)
