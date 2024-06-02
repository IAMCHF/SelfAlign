import concurrent
import mrcfile
import numpy as np
from scipy.spatial.transform import Rotation
from SelfAlign.preprocessing.img_processing import read_parameters, get_mrc_data
from scipy.ndimage import affine_transform
import numpy as N
import numpy.fft as NF
from SelfAlign.preprocessing.simulate import apply_wedge


data_temple = get_mrc_data("mask_wedge_32.mrc")
with mrcfile.open("mask_wedge_32.mrc",
                  permissive=True) as mrcData:
    mask_wedge_binned5 = mrcData.data.astype(np.float32)


def process_params(param):
    ori_path, _, quaternion, translation = param
    with mrcfile.open(ori_path) as mrc:
        volume = mrc.data.astype(np.float32)
    if isinstance(quaternion, np.ndarray):
        quaternion = Rotation.from_quat(quaternion)
    rot_matrix = quaternion.as_matrix()
    center = (np.array(volume.shape) - 1) / 2.
    offset = center - np.dot(rot_matrix, center)
    rotated_data = affine_transform(volume, rot_matrix, offset=offset + translation)
    rotated_mask = affine_transform(mask_wedge_binned5, rot_matrix, offset=offset + translation)
    result = apply_wedge(rotated_data, rotated_mask)
    return result


def average_all_parallel(settings):
    iter_count = settings.iter_count
    path_param = '{}/iter_{}/params.txt'.format(settings.result_dir, iter_count)
    params = read_parameters(path_param)
    volume_num = len(params)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_param = {executor.submit(process_params, param): param for param in params}
        results = []
        for future in concurrent.futures.as_completed(future_to_param):
            try:
                result = future.result()
            except Exception as exc:
                print(f"Exception occurred while processing {future_to_param[future]}: {exc}")
            else:
                results.append(result)

    data_sum = np.sum(results, axis=0)
    vol_sum = NF.fftshift(NF.fftn(data_sum))
    avg = N.zeros(vol_sum.shape, dtype=N.complex)
    avg = vol_sum / volume_num
    avg = N.real(NF.ifftn(NF.ifftshift(avg)))
    #
    # with mrcfile.new('{}/tmp_{}.mrc'.format(settings.result_dir, iter_count), overwrite=True) as output_mrc:
    #     output_mrc.set_data(data_temple.astype(np.float32))
    #
    with mrcfile.new('{}/refine_{}.mrc'.format(settings.result_dir, iter_count), overwrite=True) as output_mrc:
        output_mrc.set_data(avg.astype(np.float32))
    with mrcfile.new('{}/tmp_{}.mrc'.format(settings.result_dir, iter_count), overwrite=True) as output_mrc:
        output_mrc.set_data(data_temple.astype(np.float32))


def average_ini(settings):
    iter_count = settings.iter_count
    with mrcfile.new('{}/tmp_{}.mrc'.format(settings.result_dir, iter_count), overwrite=True) as output_mrc:
        output_mrc.set_data(data_temple.astype(np.float32))