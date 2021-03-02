#!/usr/bin/env python
# coding: utf-8

# __README:__
# 
# Refer to paper:
# - Liu Z, Guo Z, Ling Z, et al. Detecting Alzheimer's Disease from Speech Using Neural Networks with Bottleneck Features and Data Augmentation. ICASSP 2021.
# 
# This notebook uses bottleneck features and train a deeplearing model.
# Data augmentation is directly performed on the extracted bottleneck features.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib notebook
import matplotlib.pyplot as plt
plt.style.use('ggplot') # ggplot  seaborn-poster
# basic handling
import os
import glob
import pickle
import h5py
import numpy as np
import sklearn
# audio
import librosa
import librosa.display
import IPython.display

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

print(os.getcwd())


# # Prepare

# In[ ]:


import sys
sys.path.append('..')
from ad_detection.mlcode.util.helper_functions import *


# In[ ]:


from ad_detection.mlcode.merge_feature import merge_all

proj_root_path = '../ws_en/'
FE_file_path = proj_root_path + 'fusion/data_after_FE_CPE16.csv'


CLASS_COL_NAME = 'label'
CLASS_NAMES=("CTRL", "AD")
SEEDS = tuple(range(2008,2018))


# In[ ]:


from ad_detection.mlcode.data_splitter import KFoldSplitter

# 载入已划分的数据
data_splitter = KFoldSplitter(split_file_dir=proj_root_path + 'list/split/',
                              result_file_dir=proj_root_path + 'list/result_bottleneck/')


# In[ ]:





# # Deep Learning Dataset

# In[ ]:


get_ipython().system('ls ../ws_en/data/')
get_ipython().system('echo ----------')
get_ipython().system('ls ../ws_en/fusion/')


# In[ ]:


from ad_detection.dlcode.dl_data_manager import DLDataSet
DL_FILE_PATH = '/home/zhaoci/copy_186/bottleneck_hp.hdf5'
# '/home/zhaoci/copy_186/bottleneck_hp.hdf5'  # proj_root_path + 'fusion/bottleneck_hp.hdf5'

dl_dataset = DLDataSet(DL_FILE_PATH, FE_file_path, len(CLASS_NAMES), normlize=True,  mask_aug_num=(2,2))


# In[ ]:


dl_dataset.get_input_shape()


# In[ ]:


shape_stat = dl_dataset.describe_data()


# # 

# In[ ]:


import h5py
def length_of_sentences(shape_stat):
    global dl_dataset
    dl_dataset.get_input_shape()
    counts, bins, patch = plt.hist(shape_stat[:, 0])  # , bins=[50 * i for i in range(10)]
    for indx in range(len(counts)):
        plt.text(bins[indx], counts[indx], '%d'%counts[indx])
    plt.title('Length of Sentence')
    plt.show()
    # shape_stat[-40:, 0]

    max_length_value = np.max(shape_stat[:, 0])
    max_value_indexs = np.where(shape_stat[:, 0] == max_length_value)
    print('Max length is:', max_length_value, '\tCorresponding indexs:', max_value_indexs)
#     with h5py.File(DL_FILE_PATH, "r") as feat_clps:
#         print('Clip_id:', list(feat_clps.keys())[max_value_indexs[0][0]])
#         for indx in range(shape_stat.shape[0]):
#             if shape_stat[indx, 0] >= 3000:
#                 print(list(feat_clps.keys())[indx], end=' ')
     
length_of_sentences(shape_stat)


# In[ ]:





# In[ ]:


train_loader, test_loader, info_dict = dl_dataset.get_data_loader(2008, 2, data_splitter=data_splitter)

print('->  X shape:', info_dict['train_size'])
print('->  Y shape:', info_dict['test_size'])
print(info_dict.keys())


# In[ ]:


# 测试extract_feature是否正常
dataiter = iter(train_loader)


# In[ ]:


X_train, Y_train = next(dataiter)

def display_feature(x_i, figsize=(20,6), vmin=-3, vmax=3):
    print('x_i shape:', x_i.shape)
    plt.figure(figsize=figsize)
    librosa.display.specshow(x_i[:,:].T, sr=25, hop_length=1, x_axis='time', 
                             cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('Feature')
    plt.show()
    plt.tight_layout()

test_no = 31
x_i = X_train[test_no]
# print(info_dict['train_index'][test_no])
display_feature(x_i)
# x_i = mask_policy(x_i, mask_prob=1, num=2)
# display_feature(x_i)
print ('x_i var:', x_i.var(axis=0))
print ('x_i mean:', x_i.mean(axis=0))


# In[ ]:


# x_i_x = np.linspace(0, len(x_i)/100.0, num=len(x_i))
# plt.plot(x_i_x, x_i[:, 0], x_i_x, x_i[:, 28]+5)
# plt.show()
print('\n'.join(dl_dataset.summary()))


# In[ ]:


del X_train, Y_train, train_loader, test_loader, info_dict, dataiter, x_i

import gc
gc.collect()


# # DeepLearning Models

# In[ ]:





# In[ ]:


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
tf.__version__


# In[ ]:


# 没有问题的话就开始搭建模型
from ad_detection.dlcode.nn_model import KerasModelAdapter
from ad_detection.dlcode.model_factory_bn import fc_cnn1d_lstm_attention

UTT_LENGTH = 2560
# dl_dataset.describe_data()
dl_dataset.set_data_length(UTT_LENGTH)
print(dl_dataset.get_input_shape())


# In[ ]:


hyper_params = {
    'lr':0.001,
    'epochs':200,
    'decay_rate':0.04,
    'gpus':1,
    'batch_size':32,
    'verbose':2
}
model_creator = fc_cnn1d_lstm_attention
model = KerasModelAdapter(dl_dataset.get_input_shape(), model_creator=model_creator, **hyper_params)
print(model)
# visualize model layout with pydot_ng
model.plot_model()


# In[ ]:


# from speechemotion.mlcode.pipelineCV import PipelineCV

# pipelineCV = PipelineCV(model, dl_dataset, data_splitter, n_splits=10)
# result = pipelineCV.run_pipeline(2000)
# from speechemotion.mlcode.main_exp import gen_report, save_exp_log
# print(result['conf_mx'])
# gen_report(result['fold_metrics'])

from ad_detection.mlcode.exp_logger import Logger
logger = Logger(name_str='Bottleneck')
logger.open()
logger.log_timestamp()
logger.set_print(False, False)

model_creator_src = Logger.func2str(model_creator)
print(model_creator_src)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from ad_detection.mlcode.main_exp import main_experiment

result = main_experiment(dl_dataset, data_splitter, model, seeds=SEEDS, pipeline='dl')

conf_mx = result['conf_mx_sum']
report = result['report']


# result_df_stat # .describe()
# UAR
display(report)


# In[ ]:


from ad_detection.mlcode.main_exp import gen_report
plt.style.use('ggplot')
# show_confusion_matrix(conf_mx, save_pic_path='./log/cconf_mx.png')
plot_confusion_matrix(conf_mx, classes=CLASS_NAMES, figsize=(5,5))

logger.log_summary({
    'Memo': '|'.join(CLASS_NAMES),
    'Data': '\n%s\n' % ('\n    '.join(dl_dataset.summary())),
    'Model': '\n%s\n' % str(model),
    'Source': '\n%s\n' % model_creator_src,
    'Report': report, # gen_report(result['fold_metrics']),
    'Confusion Matrix': '\n%s\n' % repr(result['conf_mx_sum']),
    'CV_result_detail': result['cv_metrics_stat'].describe()  # fold_metrics
})


# In[ ]:


show_confusion_matrix(conf_mx)


# In[ ]:


import time
print('Stop process after 20 minutes!')
time.sleep(1200)
exit(0)


# In[ ]:




