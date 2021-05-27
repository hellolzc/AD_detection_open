#!/usr/bin/env python
import pandas as pd  # 数据分析
import numpy as np  # 科学计算
import sys
import os


####################  用于合并数据的部分  ########################


def merge_list(csv_label_path, annatation_list_file, list_dir):
    '''生成 uuids_complete_info.csv（记录合并情况），uuids_complete.txt（记录完整的数据名） '''
    uuids_have_label = pd.read_csv(csv_label_path, encoding='utf-8')
    uuids_have_annatation = pd.read_csv(annatation_list_file, encoding='utf-8', header=None, names=['uuid'])
    new_df = pd.merge(uuids_have_label, uuids_have_annatation, how='outer', on='uuid', indicator=True)
    out_file_path0 = os.path.join(list_dir, 'uuids_complete_info.csv')
    new_df.to_csv(out_file_path0, index=False)

    new_df = pd.merge(uuids_have_label, uuids_have_annatation, how='inner', on='uuid')
    out_file_path1 = os.path.join(list_dir, 'uuids_complete.txt')
    # new_df.drop(['age','education', 'sex', 'label','label_detail', 'no'], axis=1, inplace=True)
    new_df.uuid.to_csv(out_file_path1, index=False, header=False)


def merge_common(csv_label_path, data_file_path, out_file_path):
    '''duration, linguistic
    合并特征数据，根据uuid（是文件名也是被试的标识名）将特征文件和标签文件合并
    类似于数据库表的JOIN操作
    '''
    data_all = pd.read_csv(csv_label_path, encoding='utf-8')
    df = pd.read_csv(data_file_path,  encoding='utf-8')

    new_df = pd.merge(data_all, df, how='inner', on='uuid')
    print(new_df.describe())

    new_df = new_df.drop(columns=['no'])
    # 保存合并的数据
    new_df.to_csv(out_file_path, index=False)
    print('完成！')




DROP_COLUMNS = ['corpus', 'date', 'name', 'diagnosis_detail', 'no']

def check_filepath_list(filepath_list):
    new_filepath_list = []
    for fp in filepath_list:
        if os.path.exists(fp):
            new_filepath_list.append(fp)
        else:
            print('[WARN] %s does not exist!' % fp)
    return new_filepath_list

def merge_all(filepath_list, out_file_path, prefix_list=None):
    ''' 合并特征数据，根据uuid（是文件名也是被试的标识名）将多个特征文件合并
    类似于数据库表的JOIN操作
    参数:
        filepath_list : 所有需要合并的特征文件路径列表
        out_file_path : 合并后的特征文件路径
        prefix_list : 用于给一群特征加前缀
    '''
    filepath_list = check_filepath_list(filepath_list)
    print('Available Files:', filepath_list)

    df0 = pd.read_csv(filepath_list[0], encoding='utf-8')
    if prefix_list is not None:
        prefix_i = prefix_list[0]
        if prefix_i is not None:
            df0.columns = [prefix_i+x if x!='uuid' else x for x in df0.columns]

    for indx in range(1, len(filepath_list)):
        df1 = pd.read_csv(filepath_list[indx], encoding='utf-8')
        if prefix_list is not None:
            prefix_i = prefix_list[indx]
            if prefix_i is not None:
                df1.columns = [prefix_i+x if x!='uuid' else x for x in df1.columns]

        if indx == 1:
            df0 = pd.merge(df0, df1, how='inner', on='uuid')
        else:
            df0 = pd.merge(df0, df1, how='left', on='uuid')

    print(df0.head())
    print('Shape:', df0.shape)
    # 保存合并的数据
    drop_cols = set(DROP_COLUMNS).intersection(df0.columns)
    print("Drop columns:", drop_cols)
    df0 = df0.drop(columns=drop_cols)
    print("Set 'uuid' as index.")
    df0.set_index(keys='uuid', drop=True, inplace=True, verify_integrity=True)
    print('Shape:', df0.shape)
    df0.to_csv(out_file_path)
    print('记得检查数据！')


####################  用于简单预处理的部分  ########################


def _add_label_and_sex_col(data_all):
    """ sex label 数值化 因子化，返回修改后的dataframe，
    增加4列：'label_CvMvA', 'label_CvMA', 'sex_M'
    """
    # label 数值化 因子化
    dummies_label = pd.get_dummies(data_all['label'], prefix='label')
    label_cols = pd.DataFrame(columns=['label_CvMvA', 'label_CvMA'])

    label_cols.label_CvMvA = dummies_label.label_MCI + dummies_label.label_AD*2
    label_cols.label_CvMA = dummies_label.label_MCI | dummies_label.label_AD

    # add sex_M col
    dummies_sex = pd.get_dummies(data_all['sex'], prefix='sex')
    sex_M_col = dummies_sex.loc[:, 'sex_M']

    df = pd.concat([label_cols, sex_M_col, data_all], axis=1)
    print('Add cols: label_CvMvA, label_CvMA, sex_M')
    return df

def _filter_by_age_edu(data_all, age=40, edu=5):
    """ 对数据进行筛选, 限制: age>=55, education>=5
    APSIPA2019: 限制: age>=40, education>=5
    """
    print('限制: age>=%d, education>=%d' % (age, edu))

    # 丢弃55岁以下样本
    data_all = data_all[data_all.age >= age]
    # 受教育程度5年以上
    data_all = data_all[data_all.education >= edu]
    return data_all


def preprocess_steps(in_file_path, out_file_path, filter_age_edu=False):
    """ dataframe预处理，包括 标签性别因子化
    可选项: 对数据进行筛选, 限制 age>=55, education>=5
    """
    df = pd.read_csv(in_file_path, encoding='utf-8', index_col=0)
    df = _add_label_and_sex_col(df)

    if filter_age_edu:
        df = _filter_by_age_edu(df)
    print('Final data shape:', df.shape)
    df.to_csv(out_file_path)


def remove_sample_in_blacklist(ori_df_fp, blacklist_fp, output_fp):
    """ 删除黑名单里的人 """
    df = pd.read_csv(ori_df_fp, index_col=0)
    black_df = pd.read_csv(blacklist_fp, index_col=0)
    drop_uuids = black_df[black_df.discard_or_not == 1].index
    drop_uuids = df.index.intersection(drop_uuids)
    print("Drop %d rows according to '%s':\n" % (len(drop_uuids), blacklist_fp), drop_uuids)
    df.drop(index=drop_uuids).to_csv(output_fp)


if __name__ == '__main__':
    USAGE_STR = "usage: merge_feature list/all"
    args = sys.argv
    if len(args) != 2:
        print(USAGE_STR)
        exit(0)
    choose = args[1]
    # path
    proj_root_path = '../../ws_cn/'

    csv_label_path = proj_root_path + 'label/summary.csv'
    blacklist_path = proj_root_path + 'label/blacklist.csv'  # blacklist
    duration_fp = proj_root_path + 'fusion/duration.csv'
    egemaps_fp = proj_root_path + 'fusion/acoustic_CPE16.csv'
    linguistic_fp = proj_root_path + 'fusion/linguistic.csv'
    syntactic_fp = proj_root_path + 'fusion/syntactic_parse.csv'

    annatation_list_file = proj_root_path + 'list/namelist.txt'
    list_dir = proj_root_path + 'list'

    # if choose == 'duration':
    #     duration_out_fp = '../fusion/merged_dur.csv'
    #     merge_common(csv_label_path, duration_fp, duration_out_fp)
    if choose == 'all':
        # Merge different data source
        allfeature_out_fp = proj_root_path + 'fusion/merged.csv'
        merge_all([csv_label_path, duration_fp, linguistic_fp, syntactic_fp, egemaps_fp],
                  allfeature_out_fp)
        # Simple data pre-processing
        # feature_FE_fp = '../fusion/merged_FE.csv'
        preprocess_steps(allfeature_out_fp, allfeature_out_fp, filter_age_edu=True)
        if blacklist_path is not None:
            remove_sample_in_blacklist(allfeature_out_fp, blacklist_path, allfeature_out_fp)
    elif choose == 'list':
        merge_list(csv_label_path, annatation_list_file, list_dir)
    else:
        print(USAGE_STR)
