# !/usr/bin/env python
# encoding: utf-8
import mrcfile
import numpy as np


def mw2d(dim, missingAngle=[30, 30]):
    mw = np.zeros((dim, dim), dtype=np.double)
    missingAngle = np.array(missingAngle)
    missing = np.pi / 180 * (90 - missingAngle)  # [1.04719755 1.04719755], 是[60°,60°]的弧度制
    for i in range(dim):
        for j in range(dim):
            y = (i - dim / 2)
            x = (j - dim / 2)
            if x == 0:
                theta = np.pi / 2
            else:
                theta = abs(np.arctan(y / x))

            if x ** 2 + y ** 2 <= min(dim / 2, dim / 2) ** 2:
                if x > 0 and y > 0 and theta < missing[0]:
                    mw[i, j] = 1  # np.cos(theta)
                if x < 0 and y < 0 and theta < missing[0]:
                    mw[i, j] = 1  # np.cos(theta)
                if x > 0 and y < 0 and theta < missing[1]:
                    mw[i, j] = 1  # np.cos(theta)
                if x < 0 and y > 0 and theta < missing[1]:
                    mw[i, j] = 1  # np.cos(theta)

            if int(y) == 0:
                mw[i, j] = 1
    return mw


def apply_wedge(ori_data, mw3d=None):
    # ori_data数据类型是三维数组(矩阵),内容是迭代训练过程中的mrc图片经过归一化处理后的数据
    data = ori_data
    if mw3d is None:
        mw = mw2d(data.shape[1])
        outData = np.zeros(data.shape, dtype=np.float32)
        mw_shifted = np.fft.fftshift(mw)
        for i, item in enumerate(data):
            outData_i = np.fft.ifft2(mw_shifted * np.fft.fft2(item))
            outData[i] = np.real(outData_i)
        outData.astype(np.float32)
        return outData
    else:
        mw = np.fft.fftshift(mw3d)
        f_data = np.fft.fftn(ori_data)
        outData = mw * f_data
        inv = np.fft.ifftn(outData)
        outData = np.real(inv).astype(np.float32)
    return outData
