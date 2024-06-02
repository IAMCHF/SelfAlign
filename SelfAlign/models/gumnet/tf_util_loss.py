import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import math


def quaternion_to_rotation_matrix(quaternion):
    qw, qx, qy, qz = tf.unstack(quaternion, axis=-1)
    qw2, qx2, qy2, qz2 = qw * qw, qx * qx, qy * qy, qz * qz

    rotation_matrix = tf.stack([
        [1 - 2 * (qy2 + qz2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx2 + qz2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx2 + qy2)]
    ], axis=-1)

    return rotation_matrix


def quaternion_loss(y_true, y_pred):
    norm_loss = tf.reduce_sum((y_true - y_pred) ** 2, axis=-1)
    return tf.reduce_mean(norm_loss)


def translation_loss(y_true, y_pred):
    # tf.print("\ntranslation_true", y_true[0], '\n',y_true[1], '\n',y_true[3])
    # tf.print("translation_pred", y_pred[0], '\n',y_pred[1], '\n',y_pred[3])
    norm_loss = tf.reduce_sum((y_true - y_pred) ** 2, axis=-1)
    loss = tf.reduce_mean(norm_loss)
    # tf.print("translation_loss", loss, "\n")
    return loss
    # return tf.reduce_mean(tf.abs(y_true - y_pred), axis=-1)


def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    # tf.print('\ny_true', y_true[0], y_true[1], y_true[2], '\n')
    # tf.print('\ny_pred', y_pred[0], y_pred[1], y_pred[2], '\n')
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    # tf.print('\nr', r, '\n')
    return 1 - K.square(r)


def euler_angles_to_rotation_matrix(theta_z1, theta_y, theta_z2):
    Rz1 = tf.reshape([
        tf.cos(theta_z1), -tf.sin(theta_z1), 0,
        tf.sin(theta_z1), tf.cos(theta_z1), 0,
        0, 0, 1
    ], (3, 3))
    Ry = tf.reshape([
        tf.cos(theta_y), 0, tf.sin(theta_y),
        0, 1, 0,
        -tf.sin(theta_y), 0, tf.cos(theta_y)
    ], (3, 3))
    Rz2 = tf.reshape([
        tf.cos(theta_z2), -tf.sin(theta_z2), 0,
        tf.sin(theta_z2), tf.cos(theta_z2), 0,
        0, 0, 1
    ], (3, 3))
    rot_matrix = tf.matmul(Rz2, tf.matmul(Ry, Rz1))
    return rot_matrix


def rotation_matrices_from_euler(euler_angles):
    Rxyz_batch = tf.map_fn(lambda e: euler_angles_to_rotation_matrix(e[0], e[1], e[2]), euler_angles, dtype=tf.float32)
    return Rxyz_batch


def selfalign_loss(y_true, y_pred):
    batch_size = tf.shape(y_true)[0]
    # tf.print('y_pred =', y_pred.shape)
    new_first_part = (y_pred[:, :3] * 2 - 1) * tf.constant(math.pi)
    new_second_part = (y_pred[:, 3:] * 2 - 1) * 4
    # new_second_part = (y_pred[:, 3:] * 2 - 1) * 32  # test rotation and translation strategy
    scaled_y_pred = tf.concat([new_first_part, new_second_part], axis=1)
    pred_rot_euler = scaled_y_pred[:, :3]
    gt_rot_euler = y_true[:, :3]
    pred_translations = scaled_y_pred[:, 3:]
    gt_translations = y_true[:, 3:]
    pred_rot_matrices = rotation_matrices_from_euler(pred_rot_euler)  # shape: (batch_size, 3, 3)
    # tf.print('pred_rot_matrices =', pred_rot_matrices.shape)
    gt_rot_matrices = rotation_matrices_from_euler(gt_rot_euler)
    I = tf.eye(3, batch_shape=(batch_size,))
    # tf.print('I =', I.shape)
    rotation_difference = tf.matmul(tf.transpose(pred_rot_matrices, perm=[0, 2, 1]), gt_rot_matrices) - I
    rotation_loss = tf.reduce_sum(tf.square(rotation_difference), axis=[1, 2])
    translation_loss = tf.reduce_sum(tf.square(pred_translations - gt_translations), axis=1)
    total_loss = tf.reduce_mean(rotation_loss + translation_loss)
    return total_loss


def selfalign_loss_quaternion(y_true, y_pred):
    batch_size = tf.shape(y_true)[0]
    # tf.print('y_pred =', y_pred.shape)
    new_first_part = (y_pred[:, :3] * 2 - 1) * tf.constant(math.pi)
    new_second_part = (y_pred[:, 3:] * 2 - 1) * 4
    scaled_y_pred = tf.concat([new_first_part, new_second_part], axis=1)
    pred_rot_euler = scaled_y_pred[:, :3]
    gt_rot_euler = y_true[:, :3]
    pred_translations = scaled_y_pred[:, 3:]
    gt_translations = y_true[:, 3:]
    pred_rot_matrices = rotation_matrices_from_euler(pred_rot_euler)  # shape: (batch_size, 3, 3)
    # tf.print('pred_rot_matrices =', pred_rot_matrices.shape)
    gt_rot_matrices = rotation_matrices_from_euler(gt_rot_euler)
    I = tf.eye(3, batch_shape=(batch_size,))
    # tf.print('I =', I.shape)
    rotation_difference = tf.matmul(tf.transpose(pred_rot_matrices, perm=[0, 2, 1]), gt_rot_matrices) - I
    rotation_loss = tf.reduce_sum(tf.square(rotation_difference), axis=[1, 2])
    translation_loss = tf.reduce_sum(tf.square(pred_translations - gt_translations), axis=1)
    total_loss = tf.reduce_mean(rotation_loss + translation_loss)
    return total_loss
