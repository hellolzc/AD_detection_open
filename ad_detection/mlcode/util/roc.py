#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 20:02:01 2018

@author: zqguo hellolzc
"""

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt


def plot_roc(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)  
    plt.plot(fpr, tpr)
    for i in range(len(fpr)):
        if abs(fpr[i]+tpr[i]-1) < 0.01:
            print("fpr = %.2f ; tpr = %.2f" %(fpr[i], tpr[i]))
    print("AUC = %.2f" % (auc(fpr, tpr)))


def plot_scatter(x, k=242):
    plt.scatter(x[:k, 0], x[:k, 1], c='r', marker='.')
    plt.scatter(x[k:, 0], x[k:, 1], c='b', marker='.')


def get_auc(scores, labels):
    '''scores: feature value
    labels: y_true
    '''
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)  
    return auc(fpr, tpr)


def auc_classif(X, y):
    """Compute AUC stats between each feature and class.
    This score can be used to select the n_features features.
    Parameters
    ----------
    X : {matrix}, shape = (n_samples, n_features_in) Sample vectors.
    y : array-like, shape = (n_samples,) Target vector (class labels).
    Returns
    -------
    chi2 : array, shape = (n_features,) chi2 statistics of each feature.
    pval : array, shape = (n_features,) p-values of each feature.

    """
    auc_array = [get_auc(X[:, indx], y) for indx in range(X.shape[1])]
    auc_array = np.array(auc_array)
    pval = np.max(np.row_stack([auc_array, 1-auc_array]), axis=0)

    return auc_array, pval

'''
k=242
nb_mppl=np.zeros((498,2))
for i in range(498):
    nb_mppl[i][0]=nb_ppl[i,0]-nb_ppl[i,1]
    nb_mppl[i][1]=nb_ppl[i,0]+nb_ppl[i,1]
plt.scatter(nb_mppl[:k,0],nb_mppl[:k,1],c='r',marker='.',label='Control')
plt.scatter(nb_mppl[k:,0],nb_mppl[k:,1],c='b',marker='.',label='AD')    
plt.xlabel("p_diff")
plt.ylabel("p_plus")
plt.legend(loc=2)
plt.show()
'''