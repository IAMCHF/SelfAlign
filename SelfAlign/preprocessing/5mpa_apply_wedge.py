import os
import mrcfile
import numpy as np
from simulate import apply_wedge

with mrcfile.open("/HBV/Caohaofan/selfalign/mask_wedge_32.mrc", permissive=True) as mrc:
    mask_wedge_32 = mrc.data.astype(np.float32)

"snr100, snr01, snr005, snr003, snr001"
"train_data_wedge, test_data_wedge, train_data_rotation_strategy, test_data_rotation_strategy"
src_folder = '/newdata3/chf/train_data_rotation_strategy/snr001/5MPA'
dst_folder = '/newdata3/chf/train_data_rotation_strategy/snr001/5MPA_no_normalized'
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

for filename in os.listdir(src_folder):
    if filename.endswith(".mrc"):
        src_file_path = os.path.join(src_folder, filename)
        dst_file_path = os.path.join(dst_folder, filename)

        with mrcfile.open(src_file_path, permissive=True) as mrc:
            data = mrc.data.astype(np.float32)

        wedge_applied = apply_wedge(data, mask_wedge_32)

        with mrcfile.new(dst_file_path, overwrite=True) as output_mrc:
            output_mrc.set_data(wedge_applied.astype(np.float32))