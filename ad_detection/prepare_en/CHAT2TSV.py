#!/usr/bin/env python
import os

import pandas as pd
import numpy as np
from pandas import Series,DataFrame

import sys
sys.path.append('../..')
from ad_detection.prepare_en.pylangacq_modified import chat

def processOneChat(filename):
    reader=chat.SingleReader(filename)
    uuid = reader.headers()['Media'].split(',')[0] # filenames[i][:-4] + 'c'
    # Metainfo
    info_dict = reader.participants()['PAR']
    info_dict['uuid'] = uuid
    # TSV
    tsv_tuple = list(zip(*reader.utterances(clean=True, time_marker=True)))
    timemarker_tuple = list(zip(*tsv_tuple[2]))
    tsv_df = DataFrame({
        'start_time':timemarker_tuple[0],
        'end_time': timemarker_tuple[1],
        'speaker': tsv_tuple[0],
        'value': tsv_tuple[1]
    })
    tsv_df.start_time = tsv_df.start_time/1000.0
    tsv_df.end_time = tsv_df.end_time/1000.0
    return uuid, info_dict, tsv_df

def get_standard_label(group_str):
    """处理group标志，形成标准的标签"""
    if group_str == 'Control':
        return 'CTRL'
    elif group_str == 'MCI':
        return 'MCI'
    elif group_str.find('AD') != -1:
        return 'AD'
    else:
        return 'OTHER'

def get_standard_sex(sex_str):
    """处理sex标志，形成标准的标签"""
    if sex_str == 'female':
        return 'F'
    elif sex_str == 'male':
        return 'M'
    else:
        return '?'

def processCHATs(inpath, outpath, metainfo_file):
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    filenames = os.listdir(inpath)
    filenames=sorted([x for x in filenames if x.endswith('.cha')])

    metainfo_list = []
    for i in range(len(filenames)):
        filename = os.path.join(inpath, filenames[i])
        try:
            uuid, info_dict, tsv_df = processOneChat(filename)
        except Exception as e:
            print(e)
            continue
        metainfo_list.append(info_dict)
        # TSV
        filename2 = os.path.join(outpath, uuid+'.tsv')
        tsv_df = tsv_df.reindex(columns=['start_time', 'end_time', 'speaker', 'value'])
        tsv_df.to_csv(filename2, index=False, sep='\t')
    metainfo_df = DataFrame({
        'uuid': [tmpdict['uuid'] for tmpdict in metainfo_list],
        'corpus' : [tmpdict['corpus'] for tmpdict in metainfo_list],
        'age' : [tmpdict['age'].strip(';') for tmpdict in metainfo_list],
        'sex': [get_standard_sex(tmpdict['sex']) for tmpdict in metainfo_list],
        'group': [tmpdict['group'] for tmpdict in metainfo_list],
        'label': [get_standard_label(tmpdict['group']) for tmpdict in metainfo_list],
        # 'SES': [tmpdict['SES'] for tmpdict in metainfo_list],
        'score': [tmpdict['education'] for tmpdict in metainfo_list],
        # 'custom': [tmpdict['custom'] for tmpdict in metainfo_list]
    })
    metainfo_df = metainfo_df.reindex(columns=['uuid', 'corpus', 'age', 'sex', 'group', 'label', 'score'])
    metainfo_df.to_csv(metainfo_file, index=False)


def append_2_info(info1_fp, info2_fp, out_fp):
    """将CTRL AD转写文本收到的数据汇总"""
    df1 = pd.read_csv(info1_fp, encoding='utf-8')
    df2 = pd.read_csv(info2_fp, encoding='utf-8')
    df_all = pd.concat([df1, df2])
    df_all['participant_id'] = df_all['uuid'].apply(lambda x: x.split('-')[0])
    df_all.to_csv(out_fp, encoding='utf-8', index=False)


if __name__ == '__main__':
    root_path = '../../ws_en/'
    inpath = root_path + 'data_ori/ctrl/cookie/'
    outpath = root_path + 'data/tsv/'
    metainfo_file1 = root_path + 'label/info1.csv'
    processCHATs(inpath, outpath, metainfo_file1)
    inpath = root_path + 'data_ori/dementia/cookie/'
    outpath = root_path + 'data/tsv/'
    metainfo_file2 = root_path + 'label/info2.csv'
    processCHATs(inpath, outpath, metainfo_file2)
    out_fp = root_path + 'label/chat_tier_info.csv'
    append_2_info(metainfo_file1, metainfo_file2, out_fp)
    # Test:
    # inpath = '../'
    # outpath = '../'
    # metainfo_file = '../info.csv'
    # processCHATs(inpath, outpath, metainfo_file)
