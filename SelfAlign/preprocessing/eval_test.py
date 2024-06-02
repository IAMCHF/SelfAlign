import math

import numpy as np


def alignment_eval(y_true, y_pred):
    """
    y_true is defined in Radian [-pi, pi] (ZYZ convention) for rotation, and voxels for translation
    y_pred is in [0, 1] from sigmoid activation, need to scale y_pred for comparison
    """

    ang_d = []
    loc_d = []

    for i in range(len(y_true)):
        a = angle_zyz_difference(ang1=y_true[i][:3],
                                 ang2=y_pred[i][:3] * 2 * np.pi - np.pi)
        b = np.linalg.norm(
            np.round(y_true[i][3:6]) -
            np.round((y_pred[i][3:6] * 2 - 1) * 32))
        ang_d.append(a)
        loc_d.append(b)

    print('Rotation error: ', np.mean(ang_d), '+/-', np.std(ang_d),
          'Translation error: ', np.mean(loc_d), '+/-', np.std(loc_d), '----------')


# def angle_zyz_difference(ang1=np.zeros(3), ang2=np.zeros(3)):
#     rm1 = rotation_matrix_zyz(ang1)
#     rm2 = rotation_matrix_zyz(ang2)
#     trace_product = np.trace(rm1 @ rm2.T)
#     dif_d = np.arccos((trace_product - 1) / 2)
#     return dif_d


def angle_zyz_difference(ang1=np.zeros(3), ang2=np.zeros(3)):
    loc1_r = np.zeros(ang1.shape)
    loc2_r = np.zeros(ang2.shape)

    rm1 = rotation_matrix_zyz(ang1)
    rm2 = rotation_matrix_zyz(ang2)
    loc1_r_t = np.array([loc1_r, loc1_r, loc1_r])
    loc2_r_t = np.array([loc2_r, loc2_r, loc2_r])

    dif_m = (rm1.dot(np.eye(3) - loc1_r_t)).transpose() - \
            (rm2.dot(np.eye(3) - loc2_r_t)).transpose()
    dif_d = math.sqrt(np.square(dif_m).sum())

    return dif_d


def rotation_matrix_zyz(ang):
    theta_z1, theta_y, theta_z2 = ang[0], ang[1], ang[2]
    Rz1 = np.array([
        [np.cos(theta_z1), -np.sin(theta_z1), 0],
        [np.sin(theta_z1), np.cos(theta_z1), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    Rz2 = np.array([
        [np.cos(theta_z2), -np.sin(theta_z2), 0],
        [np.sin(theta_z2), np.cos(theta_z2), 0],
        [0, 0, 1]
    ])
    rot_matrix = np.dot(Rz2, np.dot(Ry, Rz1))
    return rot_matrix


def read_non_empty_lines(file_path):
    records = []
    with open(file_path, 'r') as file:
        for line in file:
            line_contents = line.strip().split()
            combined_params = np.array(line_contents[2:], dtype=np.float32)
            records.append(combined_params)
    return records

"5LQW, 5MPA, 5T2C, 6A5L"
"snr100, snr01, snr005, snr003, snr001"
"test_data, test_data_067, test_data_133, test_data_2, test_data_random, test_data_wedge, test_data_rotation_strategy"
"result_iter30, result_067, result_133, result_2, result_random, result_wedge"

y_true1 = read_non_empty_lines("/newdata3/chf/test_data_rotation_strategy/snr100/5LQW.txt")
y_pred1 = read_non_empty_lines("/newdata3/chf/result_rotation_strategy/snr100/5LQW/iter_5/params_ori.txt")
y_true2 = read_non_empty_lines("/newdata3/chf/test_data_rotation_strategy/snr01/5LQW.txt")
y_pred2 = read_non_empty_lines("/newdata3/chf/result_rotation_strategy/snr01/5LQW/iter_5/params_ori.txt")
y_true3 = read_non_empty_lines("/newdata3/chf/test_data_rotation_strategy/snr005/5LQW.txt")
y_pred3 = read_non_empty_lines("/newdata3/chf/result_rotation_strategy/snr005/5LQW/iter_5/params_ori.txt")
y_true4 = read_non_empty_lines("/newdata3/chf/test_data_rotation_strategy/snr003/5LQW.txt")
y_pred4 = read_non_empty_lines("/newdata3/chf/result_rotation_strategy/snr003/5LQW/iter_5/params_ori.txt")
y_true17 = read_non_empty_lines("/newdata3/chf/test_data_rotation_strategy/snr001/5LQW.txt")
y_pred17 = read_non_empty_lines("/newdata3/chf/result_rotation_strategy/snr001/5LQW/iter_5/params_ori.txt")

y_true5 = read_non_empty_lines("/newdata3/chf/test_data_rotation_strategy/snr100/5MPA.txt")
y_pred5 = read_non_empty_lines("/newdata3/chf/result_rotation_strategy/snr100/5MPA/iter_5/params_ori.txt")
y_true6 = read_non_empty_lines("/newdata3/chf/test_data_rotation_strategy/snr01/5MPA.txt")
y_pred6 = read_non_empty_lines("/newdata3/chf/result_rotation_strategy/snr01/5MPA/iter_5/params_ori.txt")
y_true7 = read_non_empty_lines("/newdata3/chf/test_data_rotation_strategy/snr005/5MPA.txt")
y_pred7 = read_non_empty_lines("/newdata3/chf/result_rotation_strategy/snr005/5MPA/iter_5/params_ori.txt")
y_true8 = read_non_empty_lines("/newdata3/chf/test_data_rotation_strategy/snr003/5MPA.txt")
y_pred8 = read_non_empty_lines("/newdata3/chf/result_rotation_strategy/snr003/5MPA/iter_5/params_ori.txt")
y_true18 = read_non_empty_lines("/newdata3/chf/test_data_rotation_strategy/snr001/5MPA.txt")
y_pred18 = read_non_empty_lines("/newdata3/chf/result_rotation_strategy/snr001/5MPA/iter_5/params_ori.txt")

y_true9 = read_non_empty_lines("/newdata3/chf/test_data_rotation_strategy/snr100/5T2C.txt")
y_pred9 = read_non_empty_lines("/newdata3/chf/result_rotation_strategy/snr100/5T2C/iter_5/params_ori.txt")
y_true10 = read_non_empty_lines("/newdata3/chf/test_data_rotation_strategy/snr01/5T2C.txt")
y_pred10 = read_non_empty_lines("/newdata3/chf/result_rotation_strategy/snr01/5T2C/iter_5/params_ori.txt")
y_true11 = read_non_empty_lines("/newdata3/chf/test_data_rotation_strategy/snr005/5T2C.txt")
y_pred11 = read_non_empty_lines("/newdata3/chf/result_rotation_strategy/snr005/5T2C/iter_5/params_ori.txt")
y_true12 = read_non_empty_lines("/newdata3/chf/test_data_rotation_strategy/snr003/5T2C.txt")
y_pred12 = read_non_empty_lines("/newdata3/chf/result_rotation_strategy/snr003/5T2C/iter_5/params_ori.txt")
y_true19 = read_non_empty_lines("/newdata3/chf/test_data_rotation_strategy/snr001/5T2C.txt")
y_pred19 = read_non_empty_lines("/newdata3/chf/result_rotation_strategy/snr001/5T2C/iter_5/params_ori.txt")

y_true13 = read_non_empty_lines("/newdata3/chf/test_data_rotation_strategy/snr100/6A5L.txt")
y_pred13 = read_non_empty_lines("/newdata3/chf/result_rotation_strategy/snr100/6A5L/iter_5/params_ori.txt")
y_true14 = read_non_empty_lines("/newdata3/chf/test_data_rotation_strategy/snr01/6A5L.txt")
y_pred14 = read_non_empty_lines("/newdata3/chf/result_rotation_strategy/snr01/6A5L/iter_5/params_ori.txt")
y_true15 = read_non_empty_lines("/newdata3/chf/test_data_rotation_strategy/snr005/6A5L.txt")
y_pred15 = read_non_empty_lines("/newdata3/chf/result_rotation_strategy/snr005/6A5L/iter_5/params_ori.txt")
y_true16 = read_non_empty_lines("/newdata3/chf/test_data_rotation_strategy/snr003/6A5L.txt")
y_pred16 = read_non_empty_lines("/newdata3/chf/result_rotation_strategy/snr003/6A5L/iter_5/params_ori.txt")
y_true20 = read_non_empty_lines("/newdata3/chf/test_data_rotation_strategy/snr001/6A5L.txt")
y_pred20 = read_non_empty_lines("/newdata3/chf/result_rotation_strategy/snr001/6A5L/iter_5/params_ori.txt")
#
alignment_eval(y_true1, y_pred1)
alignment_eval(y_true2, y_pred2)
alignment_eval(y_true3, y_pred3)
alignment_eval(y_true4, y_pred4)
alignment_eval(y_true17, y_pred17)

alignment_eval(y_true5, y_pred5)
alignment_eval(y_true6, y_pred6)
alignment_eval(y_true7, y_pred7)
alignment_eval(y_true8, y_pred8)
alignment_eval(y_true18, y_pred18)

alignment_eval(y_true9, y_pred9)
alignment_eval(y_true10, y_pred10)
alignment_eval(y_true11, y_pred11)
alignment_eval(y_true12, y_pred12)
alignment_eval(y_true19, y_pred19)

alignment_eval(y_true13, y_pred13)
alignment_eval(y_true14, y_pred14)
alignment_eval(y_true15, y_pred15)
alignment_eval(y_true16, y_pred16)
alignment_eval(y_true20, y_pred20)