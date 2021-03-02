#!/usr/bin/env python
# coding: utf-8

# __README:__
# 
# Refer to paper:
# - Liu Z, Guo Z, Ling Z, et al. Detecting Alzheimer's Disease from Speech Using Neural Networks with Bottleneck Features and Data Augmentation. ICASSP 2021.
# 
# This notebook repeats 10-fold cross-validation for 10 times and saves the split result.
# The split result should be generated before train any model.

# In[ ]:


# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:80% !important; }</style>"))

import os
import json
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from ad_detection.mlcode.util.helper_functions import *


# # Without Augmentation

# ## Prepare Features

# In[ ]:


from ad_detection.mlcode.merge_feature import merge_all, preprocess_steps, remove_sample_in_blacklist

proj_root_path = '../ws_en/'

# ['acoustic_CPE16.csv', 'acoustic_CPE16_lsa.csv',
# 'acoustic_IS09.csv',  'acoustic_IS09_lsa.csv',
# 'acoustic_IS10.csv',  'acoustic_IS10_lsa.csv',
# 'acoustic_IS11.csv',  'acoustic_IS11_lsa.csv', 
# 'acoustic_IS12.csv', , 'acoustic_IS12_lsa.csv',
# 'acoustic_egemaps.csv',  'acoustic_egemaps_lsa.csv',
# 'duration.csv', 'egemaps.csv', 'merged.csv', 'temp_data_after_FE.csv']


# 剔除掉不要的数据
csv_label_path = proj_root_path + 'label/summary.csv'
csv_label_selected_path = proj_root_path + 'label/summary_selected.csv'
blacklist_path = proj_root_path + 'label/blacklist.csv'  # None
remove_sample_in_blacklist(csv_label_path, blacklist_path, csv_label_selected_path)

# 合并不同的特征集
egemaps_fp = proj_root_path + 'fusion/acoustic_CPE16.csv'
allfeature_out_fp = proj_root_path + 'fusion/merged_CPE16.csv'

csv_label_path = csv_label_selected_path

merge_all([csv_label_path, egemaps_fp], 
          allfeature_out_fp,
          [None, 'CPE16_'])

# 简单数据处理
preprocess_steps(allfeature_out_fp, allfeature_out_fp)


# In[ ]:


from ad_detection.mlcode.data_manager import MLDataSet

CLASS_CHOOSE=['CTRL','AD']
SEEDS = list(range(2008, 2018))

file_path = proj_root_path + 'fusion/merged_CPE16.csv'
fe_file_path = proj_root_path + 'fusion/data_after_FE_CPE16.csv'


ad_dataset = MLDataSet(file_path)
ad_dataset.feature_engineering(class_col_name='label', class_namelist=CLASS_CHOOSE)
# ad_datasets.df = ad_datasets.drop_nan_row(ad_datasets.df, 'mmse')
ad_dataset.fill_nan_value_mean(ad_dataset.df, ['age'])

ad_dataset.write_FE_df(fe_file_path)


print()
par_id_uni = ad_dataset.df.participant_id.unique()
print('Unique %d\n' % len(par_id_uni))  # , par_id_uni
ad_dataset.df.iloc[:, 0:16].describe()


# In[ ]:


# duration egemaps linguistic score demographics    doctor all propose select test
feature_group='^CPE16_*'
# PPL_USAGE = 'origin' # None  # ['both', 'origin', 'perp']  None means auto.

ad_dataset.feature_filter(feature_regex=feature_group)


# ## Dataset Split

# In[ ]:


from ad_detection.mlcode.data_splitter import KFoldSplitter

# 重新划分数据
# 从这里开始 df里的数据顺序不能改变，否则会对不上号
k_splitter = KFoldSplitter(sampling_method=None,
                           split_file_dir='../ws_en/list/split/',
                           result_file_dir='../ws_en/list/result/')  #


# In[ ]:


k_splitter.clean()
k_splitter.split(ad_dataset.df, seeds=SEEDS)


# In[ ]:


# 3.4GHz 4 process 提一次 约 6min
# !./extract_perplexity.py
# !../code/extract_perplexity_parallel.py --process_num 20


# In[ ]:





# # Data Augmentation
# 
# ## Prepare Augmentated Features

# In[ ]:


_UUID_LENGTH = 6
def built_clps_df_from_df(input_label_fp: str, input_feature_fp: str,output_clps_label_fp: str,
                          prefix: str=None) -> None:
    """从HDF5中读取得到所有段名称，加上label.
    段的名称得符合uuid-xxxx的格式
    """
    print('input_label_fp:', input_label_fp)
    print('feature_fp:', input_feature_fp)
    print('output_clps_label_fp:', output_clps_label_fp)
    Y_df = pd.read_csv(input_label_fp)
    # Y_df = Y_df.loc[:, ('uuid', 'label', 'participant_id')]
    
    clps_df = pd.read_csv(input_feature_fp)
    if prefix is not None:
        clps_df.columns = [prefix+x if x!='uuid' else x for x in clps_df.columns]
    clps_df.rename(columns={'uuid':'clp_id'}, inplace=True)

    clps_df['uuid'] = clps_df['clp_id'].str[:_UUID_LENGTH]  # 001-0c-0000
    clps_df = pd.merge(Y_df, clps_df, on='uuid')
    #  clps_df.rename(columns={'uuid':'uuid_ori', 'clp_id':'uuid'}, inplace=True)
    clps_df.set_index('clp_id', inplace=True)
    clps_df.to_csv(output_clps_label_fp, index=True)


# In[ ]:


# %pdb
# 标签处理
proj_root_path = '../ws_en/'
csv_label_selected_path = proj_root_path + 'label/summary_selected.csv'

# 合并不同的特征集
egemaps_fp = proj_root_path + 'fusion/acoustic_CPE16_aug.csv'
allfeature_out_fp = proj_root_path + 'fusion/merged_CPE16_aug.csv'
built_clps_df_from_df(csv_label_selected_path, egemaps_fp, allfeature_out_fp, prefix='CPE16_')

# 简单数据处理
preprocess_steps(allfeature_out_fp, allfeature_out_fp)


# In[ ]:


from ad_detection.mlcode.data_manager import MLDataSet

CLASS_CHOOSE=['CTRL','AD']
SEEDS = list(range(2008, 2018))

file_path = proj_root_path + 'fusion/merged_CPE16_aug.csv'
fe_file_path = proj_root_path + 'fusion/data_after_FE_CPE16_aug.csv'


ad_dataset = MLDataSet(file_path)
ad_dataset.feature_engineering(class_col_name='label', class_namelist=CLASS_CHOOSE)
# ad_datasets.df = ad_datasets.drop_nan_row(ad_datasets.df, 'mmse')
ad_dataset.fill_nan_value_mean(ad_dataset.df, ['age'])

ad_dataset.write_FE_df(fe_file_path)


print()
par_id_uni = ad_dataset.df.participant_id.unique()
print('Unique %d\n' % len(par_id_uni))  # , par_id_uni
ad_dataset.df.iloc[:, 0:16].describe()


# In[ ]:


ad_dataset.df


# ## Dataset Split

# In[ ]:


from ad_detection.mlcode.data_splitter import KFoldSplitterAug

# 重新划分数据
# 从这里开始 df里的数据顺序不能改变，否则会对不上号
k_splitter = KFoldSplitterAug(split_file_dir= proj_root_path + 'list/split_aug/',
                              result_file_dir= proj_root_path + 'list/result_aug/')  # 


# In[ ]:


k_splitter.clean()
k_splitter.split(ad_dataset.df, seeds=SEEDS, from_split= proj_root_path + 'list/split/', ori_id_col='uuid')


# In[ ]:


train_index, test_index = k_splitter.read_split_file(2008, 0)
print(train_index, len(train_index))
print(test_index, len(test_index))


# In[ ]:





# # Data Augmentation and Sliding Window
# 
# ## Prepare Augmentated Features

# In[ ]:


get_ipython().system('cd ../ad_detection/dlcode/ && ./slice_wav_feature.py -m 3 -s 41 -j 10 -f CPE16_selected3_aug')


# In[ ]:


import h5py
hdf5_fp = '../data/acoustic_CPE16_selected3_aug_041_010.hdf5'
with h5py.File(hdf5_fp, "r") as feat_clps:
    for i, clp_id in enumerate(feat_clps):
        print(i, clp_id)
        if i > 20:
            break


# In[ ]:


# %pdb
# 标签处理
proj_root_path = '../'

CLASS_CHOOSE=['CTRL','AD']
SEEDS = list(range(2008, 2018))


fe_file_path = '../fusion/data_after_FE_CPE16.csv'
aug_h5_path = '../data/acoustic_CPE16_selected3_aug_041_010.hdf5'
out_df_path = '../fusion/data_after_FE_CPE16_aug_041_010.csv'

# 给新的 hdf5 文件生成 label
# print('../ad_detection/dlcode/gen_label_for_aug.py -i %s  -f %s  -o %s' % (fe_file_path, aug_h5_path, out_df_path))

from ad_detection.dlcode.gen_label_for_aug import built_clps_df
built_clps_df(fe_file_path, aug_h5_path, out_df_path)


# In[ ]:





# In[ ]:


import pandas as pd
df = pd.read_csv('../fusion/data_after_FE_CPE16_aug_041_010.csv', index_col=0)
df.head()


# ## Dataset Split

# In[ ]:


from ad_detection.mlcode.data_splitter import KFoldSplitterAug

# 重新划分数据
# 从这里开始 df里的数据顺序不能改变，否则会对不上号
k_splitter = KFoldSplitterAug(split_file_dir='../list/split_clps/',
                              result_file_dir='../list/result_clps/',
                              test_on_origin=False)


# In[ ]:


k_splitter.clean()
k_splitter.split(df, seeds=SEEDS, from_split='../list/split/', ori_id_col='uuid')


# In[ ]:


train_index, test_index = k_splitter.read_split_file(2008, 0)
print(train_index, len(train_index))
print(test_index, len(test_index))


# In[ ]:





# In[ ]:




