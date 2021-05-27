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





# In[ ]:





# In[ ]:




