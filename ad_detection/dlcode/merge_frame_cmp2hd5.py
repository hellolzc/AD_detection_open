#!/usr/bin/env python
# -*- coding: utf-8 -*-
# hellolzc 20200716
""" 用于处理Bottleneck特征文件, 将保存到HDF5文件中
这里处理的都是时序特征(LLDs)
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse
import h5py


import numpy as np

def load_mat_float_htk(filename):
    fid = open(filename, 'rb')
    nSamples = np.fromfile(fid, '>i', 1)
    sampPeriod = np.fromfile(fid, '>i', 1)
    sampSize = np.fromfile(fid, '>h', 1) // 4
    paraKind = np.fromfile(fid, '>h', 1)

    data = np.fromfile(fid, '>f4').reshape(nSamples[0], sampSize[0])
    fid.close()
    return data


mean_invstd = None  # mean and inv std

def load_norm_data(norm_data_path:str) -> None:
    global mean_invstd
    mean_invstd = np.loadtxt(norm_data_path).reshape((2, 514))

def remove_duplicate_norm(in_data: np.array, remove_time_info=True) -> np.array:
    """remove duplicated frame and time information"""
    global mean_invstd
    in_data = in_data[::8, :].copy()
    in_data = (in_data - mean_invstd[0][:]) * mean_invstd[1][:]
    if remove_time_info:
        in_data = in_data[:, 2:]
    return in_data


def single_LLDs(file_path):
    """ 载入一个时序特征文件，返回一个numpy array
    """
    # sampling_frequency = 100
    data = load_mat_float_htk(file_path)
    data = remove_duplicate_norm(data)
    return data


def single_LLDs_main(input_dir, output_file, norm_file):
    """ LLDs（low level descriptors）LLDs指的是手工设计的一些低水平特征，一般是在一帧语音上进行的计算，是用来表示一帧语音的特征。
        HSFs（high level statistics functions）是在LLDs的基础上做一些统计而得到的特征，是用来表示一个utterance的特征。
    将保存到HDF5文件中"""

    # print('seg_time %f hop_time %f' % (feature_type, start_time, end_time))
    print('inputdir:', input_dir)
    print('outfile:', output_file)
    print('norm_file:', norm_file)

    load_norm_data(norm_file)

    file_list = os.listdir(input_dir)
    file_list = [fp for fp in file_list if fp[-4:]=='.cmp']
    file_list.sort()

    with h5py.File(output_file,"w") as h5file:
        for line in file_list:
            uuid = line[:-4]  # remove '.cmp'
            print(uuid, end=' ', flush=True)
            feature_path = os.path.join(input_dir, line)

            merged_LLD = single_LLDs(feature_path)
            h5file.create_dataset(uuid, data=merged_LLD)
    print()



if __name__ == '__main__':
    # e.g.
    # ./merge_frame_feature.py -c feature_selected_config_1.json
    parser = argparse.ArgumentParser()
    # norm_data_path = '../ws_en/data/bottleneck/input.norm'
    parser.add_argument(
        '-c', '--norm_file',
        type=str,
        default='../../ws_en/data/bottleneck/input.norm',
        help='The name of the config file. Ignore other arguments if this one is set.'
    )
    parser.add_argument(
        '-i', '--input_dir',
        type=str,
        default='../../ws_en/data/bottleneck/hp_aug/',
        help='The path of bottleneck feature'
    )
    parser.add_argument(
        '-o', '--output_file',
        type=str,
        default='../../ws_en/fusion/bottleneck_hp_aug.hdf5',
        help='The name of the out put hdf5 file'
    )

    FLAGS, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print('Unknown arguments: ', unparsed)
    args = sys.argv

    single_LLDs_main(FLAGS.input_dir, FLAGS.output_file, FLAGS.norm_file)


