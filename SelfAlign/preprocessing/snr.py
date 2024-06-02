import os
import shutil
import numpy as np
import mrcfile
from normalize_z_score_file import normalize_z_score_single_file


def add_gaussian_noise(input_folder, output_folder, snr):
    noise_std = 1.0 / snr
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith(".mrc"):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)
            with mrcfile.open(input_filepath, 'r') as mrc:
                data = mrc.data.astype(np.float32)
            noise = np.random.normal(scale=noise_std, size=data.shape)
            noisy_data = data + noise
            with mrcfile.new(output_filepath, overwrite=True) as mrc:
                mrc.set_data(noisy_data.astype(np.float32))

def correlation_coefficient(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = np.mean(x)
    my = np.mean(y)
    xm, ym = x - mx, y - my
    r_num = np.sum(np.multiply(xm, ym))
    r_den = np.sqrt(np.multiply(np.sum(np.square(xm)), np.sum(np.square(ym))))
    r = r_num / r_den
    return r



"5LQW, 5MPA, 5T2C, 6A5L"
"snr01, snr005, snr003, snr001"
add_gaussian_noise("/newdata3/chf/normalized/snr100/6A5L",
                   "/newdata3/chf/noise/snr001/6A5L", 3)

"5LQW, 5MPA, 5T2C, 6A5L"
"snr01, snr005, snr003, snr001"
normalize_z_score_single_file("/newdata3/chf/noise/snr001/6A5L/6A5L.mrc",
                              "/newdata3/chf/normalized/snr001/6A5L/6A5L.mrc")

with mrcfile.open("/newdata3/chf/normalized/snr100/6A5L/6A5L.mrc", 'r') as mrc:
    data1 = mrc.data.astype(np.float32)
with mrcfile.open("/newdata3/chf/noise/snr001/6A5L/6A5L.mrc", 'r') as mrc:
    data2 = mrc.data.astype(np.float32)
c = correlation_coefficient(data1, data2)
print("snr = ", c / (1 - c))


# with mrcfile.open("/newdata3/chf/normalized/snr100/5LQW/5LQW.mrc", 'r') as mrc:
#     data1 = mrc.data.astype(np.float32)
# with mrcfile.open("/newdata3/chf/normalized/snr100/5LQW/rotated_0.mrc", 'r') as mrc:
#     data2 = mrc.data.astype(np.float32)
# c = correlation_coefficient(data1, data2)
# print("snr = ", c / (1 - c))