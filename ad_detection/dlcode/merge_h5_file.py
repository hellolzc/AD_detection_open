#!/usr/bin/env python
# -*- coding: utf-8 -*-
# hellolzc 20200419
""" Merge HDF5 files, which store LLDs
Notes from OpenSMILE toolkit:
LLDs（low level descriptors）LLDs指的是手工设计的一些低水平特征，一般是在一帧语音上进行的计算，是用来表示一帧语音的特征。
HSFs（high level statistics functions）是在LLDs的基础上做一些统计而得到的特征，是用来表示一个utterance的特征。
"""

import os, sys, argparse
from typing import List
import numpy as np
import pandas as pd
import h5py


def merge_h5_main(root_dir: str, output_fp: str, input_fp_list: List[str]):
    """ 合并HDF5文件 """
    output_file = os.path.join(root_dir, output_fp)

    print("InputConfig:")
    for item in input_fp_list:
        print('\t', item)
    print("Output:", output_file)
    print()

    with h5py.File(output_file,"w") as h5file:
        for line in input_fp_list:
            input_file = os.path.join(root_dir, line)
            with h5py.File(input_file, 'r') as h5_input:
                for key in h5_input:
                    feat_data = h5_input[key]
                    h5file.create_dataset(key, data=feat_data)

        print("\nLast sample:", key, "shape:", h5file[key].shape)



if __name__ == '__main__':
    # e.g.
    # ./merge_h5_file.py -r ../../data -o acoustic_selected3_ori_aug.hdf5 -i "acoustic_CPE16_selected3.hdf5" "acoustic_CPE16_selected3_aug.hdf5"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        default="../../data",
        help='The path of root directory.'
    )
    parser.add_argument(
        '-o', '--output_file',
        type=str,
        default="acoustic_selected3_ori_aug.hdf5",
        help='The name of the config file. Ignore other arguments if this one is set.'
    )
    parser.add_argument(
        '-i', "--inputs",
        nargs="*",  # expects ≥ 0 arguments
        type=str,
        default=["acoustic_CPE16_selected3.hdf5", "acoustic_CPE16_aug_selected3.hdf5"],
        help='Input file list'
    )

    FLAGS, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print('Unknown arguments: ', unparsed)

    merge_h5_main(FLAGS.root_dir, FLAGS.output_file, FLAGS.inputs)


