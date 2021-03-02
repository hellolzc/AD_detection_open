""" Compared with `PipelineCV`, DLPipelineCV use `DataGenerator` instead of
    loading all data to the memory (May be slower due to keras implementaion)
"""
import os
import gc
import numpy as np
import pandas as pd
import time

import sklearn
from sklearn.metrics import confusion_matrix
from ad_detection.mlcode.util.helper_functions import accuracy, precision_recall_f1score
from ad_detection.mlcode.data_splitter import DataSplitter
from ad_detection.mlcode.pipelineCV import PipelineBaseClass

from ad_detection.dlcode.dl_data_manager import DLDataSet
from ad_detection.dlcode.nn_model import KerasModelAdapter


class DLPipelineCV(PipelineBaseClass):
    """ Make K-Fold Cross-validation: analysis and save the result of each fold."""
    def __init__(self, model_base : KerasModelAdapter, dataset : DLDataSet,
                 data_splitter : DataSplitter,
                 exp_name='untitled'):
        super(DLPipelineCV, self).__init__()
        self.model_base = model_base  # Model
        self.n_splits = data_splitter.n_splits
        self.exp_name = exp_name
        self.class_num = dataset.class_num
        self.dataset = dataset  # DLDataSet
        self.data_splitter = data_splitter

    def log_result(self, train_pred, train_true, test_pred, test_true, model : KerasModelAdapter=None):
        """save result for future analysising"""
        fold_i_result = {
            'train_pred': DataSplitter.array2CSstr(train_pred),
            'train_true': DataSplitter.array2CSstr(train_true),
            'test_pred': DataSplitter.array2CSstr(test_pred),
            'test_true': DataSplitter.array2CSstr(test_true),
        }
        # 记录模型参数
        if model:
            model_params = model.log_parameters()
            fold_i_result = dict(fold_i_result, **model_params)
        return fold_i_result

    def one_fold_in_CV(self, seed, ith):
        """ 十折交叉验证中的一折
        返回：fold_i_result, fold_i_stat, conf_mx
        """
        train_generator, test_generator, info_dict = self.dataset.get_data_loader(seed, ith,
            data_splitter=self.data_splitter)
        train_set_size = info_dict['train_size']
        test_set_size = info_dict['test_size']
        train_batch_size = info_dict['train_batch_size']
        test_batch_size =  info_dict['test_batch_size']
        Y_train = info_dict['Y_train']
        Y_test = info_dict['Y_test']

        model = self.model_base.clone_model()
        # 训练
        assert model.batch_size == train_batch_size
        model.fit_generator(train_generator, test_generator,
            train_set_size=train_set_size, val_set_size=test_set_size)
        # save model
        model.save_model(self.data_splitter.get_result_file_dir() + '/model_%d_%d.h5' % (seed, ith))

        # 预测  重置生成器
        train_generator, test_generator, info_dict = self.dataset.get_data_loader(seed, ith,
            data_splitter=self.data_splitter, shuffle=False)
        train_pred = model.predict_generator(train_generator, train_set_size, train_batch_size)
        test_pred = model.predict_generator(test_generator)
        # 记录结果
        fold_i_result = self.log_result(train_pred, Y_train, test_pred, Y_test, model)
        # 记录评估
        fold_i_stat = self.evaluate(train_pred, Y_train, test_pred, Y_test)

        del train_generator, test_generator

        return fold_i_result, fold_i_stat


    def run_pipeline(self, seed):
        """ 一次交叉验证，对X，Y训练模型，返回结果的字典，包含DataFrame
        预测的标签由数据集中的label_group指定
        """
        fold_metrics = pd.DataFrame(columns=['train_acc', 'test_acc',
                                        'train_precision', 'train_recall', 'train_f1score',
                                        'test_precision', 'test_recall', 'test_f1score'])

        k_fold_results = {}
        for ith in range(self.n_splits):

            fold_i_result, fold_i_stat = self.one_fold_in_CV(seed, ith)
            # garbage collection
            gc.collect()
            # print log
            timestr = time.strftime('%H:%M:%S', time.localtime(time.time()))
            print('Seed: %d, Fold: %d, Time: %s' % (seed, ith, timestr), end='\t')
            print('Acc: Train %f, Test %f' % (fold_i_stat['train_accuracy'], fold_i_stat['test_accuracy']))

            conf_mx_i = fold_i_stat['conf_mx']
            if ith == 0:
                conf_mx = conf_mx_i
            else:
                conf_mx += conf_mx_i

            k_fold_results[ith] = fold_i_result
            fold_metrics.loc[ith] = [
                fold_i_stat['train_accuracy'],
                fold_i_stat['test_accuracy'],
                fold_i_stat['train_precision'],
                fold_i_stat['train_recall'],
                fold_i_stat['train_f1score'],
                fold_i_stat['test_precision'],
                fold_i_stat['test_recall'],
                fold_i_stat['test_f1score']
            ]
        print('========Seed:%d========'%seed)
        print('\tTest Acc:', fold_metrics['test_acc'].mean())
        print('=========================\n')
        self.data_splitter.save_result(k_fold_results, seed, self.exp_name)

        result = {
            'fold_metrics': fold_metrics,
            'conf_mx': conf_mx
        }
        return result

