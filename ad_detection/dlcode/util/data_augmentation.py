from typing import List, Tuple
import math
import numpy as np


def channel_masking_(data, mask_prob=1, f_max=None, factor=1):
    yes_or_not = np.random.random() < mask_prob
    if not yes_or_not:
        return
    timestep, channel = data.shape
    if f_max is None:
        f_max = np.round(channel / 10.0 * factor)
    f_m = int(np.round(np.random.random() * f_max))
    f_mask = np.random.permutation(channel)[:f_m]
    if f_m > 0:
        data[:, f_mask] = 0
    return


def frequency_masking_(data, mask_prob=1, f_max=None):
    yes_or_not = np.random.random() < mask_prob
    if not yes_or_not:
        return
    timestep, channel = data.shape
    if f_max is None:
        f_max = np.round(channel / 10.0)
    f_m = np.round(np.random.random() * f_max)
    f_0 = np.round(np.random.random() * (channel - f_m))
    if f_m > 0:
        data[:, int(f_0) : int(f_0+f_m)] = 0
    return


def time_masking_(data, mask_prob=1, t_max=25):
    yes_or_not = np.random.random() < mask_prob
    if not yes_or_not:
        return
    timestep, channel = data.shape

    t_m = np.round(np.random.random() * t_max)
    t_0 = np.round(np.random.random() * (timestep - t_m))
    if t_m > 0:
        data[int(t_0) : int(t_0+t_m)] = 0
    return


def mask_policy(data, mask_prob=0.7, t_max=25, f_max=None, num=1, inplace=True):
    """ 用遮盖的方式做数据增强
    num为整数时表示 frequency_mask time_masking 增强的次数
    num为list事表示 三种增强方式各自的增强次数
    """
    if not inplace:
        data = data.copy()
    yes_or_not = np.random.random() < mask_prob
    if not yes_or_not:
        return data
    if isinstance(num, int):
        for _ in range(num):
            frequency_masking_(data, f_max=f_max)
            time_masking_(data, t_max=t_max)
    else:
        assert isinstance(num, (List, Tuple))
        for _ in range(num[0]):
            frequency_masking_(data, f_max=f_max)
        for _ in range(num[1]):
            time_masking_(data, t_max=t_max)
        if len(num)==3:
            channel_masking_(data, f_max=f_max, factor=num[2])
    return data
