#!/usr/bin/env python
import pandas as pd  # 数据分析
import numpy as np  # 科学计算
import time
import sklearn

from typing import List
from ad_detection.mlcode.data_manager import DataSet
from ad_detection.mlcode.data_splitter import DataSplitter
from ad_detection.mlcode.model_base_class import Model
from ad_detection.mlcode.pipelineCV import PipelineCV
from ad_detection.mlcode.exp_logger import Logger


def gen_report(result_df_stat):
    """ 生成一个方便粘贴到实验报告里的表格 """
    result_report = result_df_stat.describe()
    mean = result_report.loc['mean',:]
    report = pd.DataFrame(data=[[mean['train_acc'],mean['train_precision'],mean['train_recall'],mean['train_f1score']],
                 [mean['test_acc'],mean['test_precision'],mean['test_recall'],mean['test_f1score']]], 
                 index=['train','test'], 
                 columns=['accuracy','precision','recall','F1_score'])
    return report

# def save_exp_log(report_dict, name_str='', float_format='%.4f', output_dir='./log/'):
#     """ 保存实验报告，记录实验时间和结果到文件 ./log/%Y-%m-%d_name_str.log,
#     同一天同一个name_str的实验结果会追加到同一个文件中
#     report_dict是一个字典，key是str，value必须可以转成str或者是DataFrame
#     """
#     date_str = time.strftime('%Y-%m-%d', time.localtime(time.time()))
#     timestr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
#     file_name = output_dir + date_str + '_' + name_str + '.log'
#     with open(file_name, mode='a') as f:
#         f.write('\n' + '====' * 20 + '\n')
#         f.write('Time: %s\n' % timestr)
#         for key in report_dict:
#             f.write('%s:' % key)
#             value = report_dict[key]
#             if type(value) == pd.DataFrame:
#                 f.write('\n')
#                 value.to_csv(f, sep='\t', float_format=float_format)
#             else:
#                 f.write(str(value))
#             f.write('\n')
#         f.write('\n')

def main_experiment(ad_dataset: DataSet, data_splitter: DataSplitter,
                    model: Model, seeds: List[int], pipeline='ml', exp_name='untitled'):
    """ 实验主过程: 多次调用pipelineCV之后取平均
    seeds 我常用 list(range(1998, 2018)), seeds的长度决定十折的次数，必须跟划分数据时所用的一致
    """
    n_splits = data_splitter.n_splits
    exp_logger = Logger.get_instance()

    print(model)

    # 用来记录十折每一折的结果
    fold_metrics_stat = pd.DataFrame(columns=['train_acc', 'test_acc','train_precision', 'train_recall', 'train_f1score',
                                          'test_precision', 'test_recall', 'test_f1score'], dtype=float)
    # 用来记录每次十折的平均结果
    cv_metrics_stat = pd.DataFrame(columns=['train_acc', 'test_acc','train_precision', 'train_recall', 'train_f1score',
                                          'test_precision', 'test_recall', 'test_f1score'], dtype=float)
    if pipeline == 'ml':
        pipelineCV = PipelineCV(model, ad_dataset, data_splitter, exp_name)
    else:
        assert pipeline == 'dl'
        from ad_detection.dlcode.dl_pipeline import DLPipelineCV
        pipelineCV = DLPipelineCV(model, ad_dataset, data_splitter, exp_name)
    for indx in range(len(seeds)):
        seed = seeds[indx]
        exp_logger.log_timestamp()
        exp_logger.log_detail('@seed: ' + str(seed))

        # run
        result = pipelineCV.run_pipeline(seed)
        fold_metrics = result['fold_metrics']
        conf_mx = result['conf_mx']
        if indx == 0:
            conf_mx_sum = conf_mx
        else:
            conf_mx_sum += conf_mx
        for i in range(n_splits):
            fold_metrics_stat.loc[indx*n_splits + i] = fold_metrics.loc[i]
        cv_metrics_stat.loc[indx] = fold_metrics.mean(axis=0)

        # show_confusion_matrix(conf_mx)
        exp_logger.log_detail('@confusion_matrix:\n' + repr(conf_mx))
        exp_logger.log_detail('@fold_metrics:\n' + exp_logger.dataframe2TSVstr(fold_metrics))
        exp_logger.log_detail('@fold_metrics_stat:\n' + exp_logger.dataframe2TSVstr(fold_metrics.describe()))
    result = {
        'fold_metrics_stat': fold_metrics_stat,
        'cv_metrics_stat': cv_metrics_stat,
        'conf_mx_sum': conf_mx_sum,
        'report': gen_report(cv_metrics_stat)
    }

    return result
