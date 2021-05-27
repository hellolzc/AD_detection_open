#!/usr/bin/env python
# coding: utf-8

# __README:__
# 
# Refer to paper:
# - Liu Z, Guo Z, Ling Z, et al. Detecting Alzheimer's Disease from Speech Using Neural Networks with Bottleneck Features and Data Augmentation. ICASSP 2021.
# 
# This notebook uses HSFs extracted with OpenSMILE and train a support vector machines (SVM) as the classifier.

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


# In[ ]:





# # Prepare Features

# In[ ]:


from ad_detection.mlcode.data_manager import MLDataSet

proj_root_path = '../'
fe_file_path = proj_root_path + 'fusion/data_after_FE_CPE16.csv'


CLASS_CHOOSE=['CTRL','AD']
SEEDS = list(range(2008, 2018))

ad_dataset = MLDataSet(fe_file_path)

print()
par_id_uni = ad_dataset.df.participant_id.unique()
print('Unique %d\n' % len(par_id_uni))  # , par_id_uni
ad_dataset.df.iloc[:, 0:16].describe()


# In[ ]:





# In[ ]:


# duration egemaps linguistic score demographics    doctor all propose select test
feature_group='^CPE16_*'
# ['both', 'origin', 'perp']  None means auto.
PPL_USAGE = 'origin' # None

ad_dataset.feature_filter(feature_regex=feature_group)
# ad_dataset.ppl_usage('both')


# # Dataset Split

# In[ ]:


from ad_detection.mlcode.data_splitter import KFoldSplitter

# 载入已划分的数据
k_splitter = KFoldSplitter(split_file_dir='../list/split/', result_file_dir='../list/result/') # 


# In[ ]:


from ad_detection.mlcode.exp_logger import Logger
logger = Logger(name_str=feature_group.strip('^*'))
logger.open()
logger.log_timestamp()
logger.set_print(False, False)


# # Train & Test

# In[ ]:


import sklearn

from sklearn import linear_model, decomposition, datasets
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

from ad_detection.mlcode.util.roc import auc_classif
from ad_detection.mlcode.util.feat_selection import SBS


# In[ ]:


# %pdb
#################################################

# gamma: 当kernel为‘rbf’, ‘poly’或‘sigmoid’时的kernel系数。
# 如果不设置，默认为 ‘auto’ ，此时，kernel系数设置为：1/n_features
# C: 误差项的惩罚参数，一般取值为10的n次幂，如10的-5次幂，10的-4次幂。。。。10的0次幂，10，1000,1000，在python中可以使用pow（10，n） n=-5~inf
#     C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，这样会出现训练集测试时准确率很高，但泛化能力弱。
#     C值小，对误分类的惩罚减小，容错能力增强，泛化能力较强。

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]}]

svc_model =sklearn.svm.SVC()
# # parameters = {'kernel':('linear', 'rbf'), 'C':[1, 2, 4], 'gamma':[0.125, 0.25, 0.5 ,1, 2, 4]}
# model = GridSearchCV(svc_model, tuned_parameters[0], scoring='f1', cv=4)


# fit到RandomForestRegressor之中
# C = [10, 5, 1, 0.7, 0.5, 0.2, 0.1, 0.05, 0.01]
# model = linear_model.LogisticRegression(C=1.0, penalty='l1', solver='liblinear')  # C=0.1, tol=1e-6 lbfgs
# model = sklearn.svm.SVC(kernel='linear', C=0.1)
# model = RandomForestClassifier(n_estimators=50, max_leaf_nodes=5) # , max_features=5, max_features=10, max_depth=None
# model = linear_model.LogisticRegression(C=0.1, penalty='l1', solver='liblinear')
# model = KNeighborsClassifier(n_neighbors=10)
# model = linear_model.LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], penalty='l1', solver='liblinear', cv=4, max_iter=1000)  # , 100

# model = MLPClassifier(solver='sgd', hidden_layer_sizes = (100,30), random_state = 1, max_iter=500)

model_list = {
#     'svm1':sklearn.svm.SVC(kernel='rbf', gamma=1e-4, C=10),
#     'svm2':sklearn.svm.SVC(kernel='linear', C=0.01),
    'svm1':GridSearchCV(svc_model, tuned_parameters[0], scoring='f1', cv=4),
    'svm2':GridSearchCV(svc_model, tuned_parameters[1], scoring='f1', cv=4),
#     'lr1':linear_model.LogisticRegressionCV(Cs=[0.01, 0.1, 1], penalty='l1', solver='liblinear', cv=4),
#     'rf1':RandomForestClassifier(n_estimators=50, max_leaf_nodes=5)
}



# lsvc = # LinearSVC(C=0.01, penalty="l1", dual=False)
# l1norm_model = linear_model.LogisticRegression(C=0.1, penalty='l1')
# pca = decomposition.PCA(10)

# model = Pipeline([
#     # SBS(model, 15)
#     # SelectKBest(mutual_info_classif, k=5)   # auc_classif
#     # SelectKBest(auc_classif, k=10)
#     # SelectFromModel(model, threshold="0.1*mean")
#   ('feature_selection', SelectKBest(auc_classif, k=3) ),
#   ('classification', model)
# ])
# model = Pipeline(steps=[('pca', pca), ('clf', model)])


# In[ ]:


# %pdb
from ad_detection.mlcode.mlmodel import SKLearnModelAdapter
from ad_detection.mlcode.main_exp import main_experiment


# In[ ]:


for key  in model_list:
    model = model_list[key]
    adapter = SKLearnModelAdapter(model)
    result = main_experiment(ad_dataset, k_splitter, adapter, seeds=SEEDS)

    conf_mx = result['conf_mx_sum']
    report = result['report']

    show_confusion_matrix(conf_mx, save_pic_path='./log/cconf_mx.png', figsize=(3,3))

    # result_df_stat # .describe()
    # UAR
    print(report)

    logger.log_summary({
        'Memo': '|'.join(CLASS_CHOOSE),
        'Data': ad_dataset.summary(),
        'Model': '\n%s\n' % str(model),
        'Report': report,
        'Confusion Matrix': '\n%s\n' % repr(result['conf_mx_sum']),
        'CV_result_detail': result['cv_metrics_stat'].describe()
    })


# In[ ]:


logger.close()


# # Analysis

# ## Plot learning curve

# In[ ]:


# model = linear_model.LogisticRegression(C=0.1, penalty='l1')
# model = RandomForestClassifier(n_estimators=50, max_leaf_nodes=5)
model = sklearn.svm.SVC(kernel='linear', C=0.01)
import warnings
warnings.filterwarnings("ignore", message="The default value of cv will change from 3 to 5 in version 0.22.")

X, Y = ad_dataset.get_XY() # , return_matrix=True
print(X.columns)
# randomforest和logisticRegression已知对变量数量级和变化范围不敏感

X_train, X_test, Y_train, Y_test, _ = ad_dataset.get_data_scaled(2009, 3, scale=True, data_splitter=k_splitter)
print(model)
print(X.shape)  # _train
plot_learning_curve(model, "Learning Curve", X_train, Y_train, train_sizes=np.linspace(0.2, 1.0, 5), ylim=(0.5,1.0))
plt.show()


# In[ ]:





# # Try ...

# In[ ]:





# In[ ]:





# In[ ]:




