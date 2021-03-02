import numpy as np
import pandas as pd


def read_perp(seed, fold_no):
    '''load perplexity file
    perp_path = '../ws_cn/data/perp/perp_%d_%d.csv' % (seed, fold_no)
    return df. cols: 'perp_c0', 'perp_c1', 'perp_c2', ...
    '''
    perp_path = '../ws_cn/data/perp/perp_%d_%d.csv' % (seed, fold_no)
    df = pd.read_csv(perp_path, index_col='uuid')
    test_str = df.value[0]
    component_num = len(test_str.split(':'))
    assert component_num in [2, 3]

    for comp_i in range(component_num):
        col_name = 'perp_c%d' % comp_i
        df[col_name] = df['value'].map(lambda x: float(x.split(':')[comp_i]))

    if component_num == 2:
        df['perp_c2'] = df['perp_c1'] - df['perp_c0']
    else:
        df['perp_c3'] = df['perp_c1'] - df['perp_c0']
        df['perp_c4'] = df['perp_c2'] - df['perp_c0']
        df['perp_c5'] = df['perp_c2'] - df['perp_c1']

    return df


def combine_perp(X_df, seed, ith, ppl_usage, data_splitter):
    ''' 将X,Y划和perplexity拼接起来
    ppl_usage = ['both', 'perp', 'origin']
    '''
    _, test_index = data_splitter.read_split_file(seed, ith)

    if ppl_usage != 'origin':
        perp_df = read_perp(seed, ith)
        # print(X_df.head())
        assert (perp_df.loc[test_index].flag.values.any() == 1)

        filter_regex = r'perp_c\d'  # '|'.join(['perp_c0', 'perp_c1'])
        tmp_df = perp_df.filter(regex=filter_regex)

    #
    if ppl_usage == 'both':
        X_df = pd.merge(X_df, tmp_df, left_index=True, right_index=True)  # , on='uuid'
    elif ppl_usage == 'perp':
        X_df = tmp_df
    elif ppl_usage == 'origin':
        X_df = X_df
    else:
        print('Unknown Option %s' % ppl_usage)
        X_df = X_df
    return X_df