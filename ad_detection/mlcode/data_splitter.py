import pandas as pd
import numpy as np
import random
import json
import os, time
from typing import Tuple, List, Dict

from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from .util.stratifiedGroupKFold import RepeatedStratifiedGroupKFold

# os.path.join(os.path.dirname(__file__), '../../list/split/')
_DEFAULT_SPLIT_FILE_HOME_ = './exp/list/split/'
_DEFAULT_RESULT_FILE_HOME_ = './exp/list/result/'


class DataSplitter(object):
    """基类, 用于确定接口. 处理关于划分数据集的类"""
    def __init__(self, n_splits: int=10, split_file_dir: str=None, result_file_dir: str=None):
        """  处理关于划分数据集的类 基类初始化  """
        self.n_splits = n_splits
        self.seeds = []
        if split_file_dir is None:
            self.split_file_dir = _DEFAULT_SPLIT_FILE_HOME_
        else:
            self.split_file_dir = split_file_dir
        if result_file_dir is None:
            self.result_file_dir = _DEFAULT_RESULT_FILE_HOME_
        else:
            self.result_file_dir = result_file_dir
        self._check_directory(self.result_file_dir)
        self._check_directory(self.split_file_dir)

    @staticmethod
    def array2CSstr(result_array: np.ndarray) -> str:
        """convert result 1-d array to comma separated string"""
        result_list = [str(val) for val in list(result_array)]
        return ','.join(result_list)

    @staticmethod
    def CSstr2array(result_str: str) -> np.ndarray:
        """convert comma separated string to 1-d array"""
        result_list = result_str.split(',')
        return np.array([float(val) for val in result_list])

    @staticmethod
    def _check_directory(dir_path: str):
        if not os.path.exists(dir_path):
            print("[INFO @ %s]"%__name__, "Make directory: %s" % dir_path)
            os.makedirs(dir_path)

    @staticmethod
    def _delete_files(dir_path: str, file_ext: str):
        for files in os.listdir(dir_path):
            if files.endswith(file_ext):  # ".json"
                os.remove(os.path.join(dir_path, files))

    def clean(self, split=True, result=False):
        """清理已存在的分割文件, 方便重新开始"""
        self._check_directory(self.result_file_dir)
        self._check_directory(self.split_file_dir)
        # 删除所有的json文件
        if split:
            self._delete_files(self.split_file_dir, '.json')
        if result:
            self._delete_files(self.result_file_dir, '.json')
        
    
    def split(self, df: pd.DataFrame, seeds: List[int]):
        """ 对df做K折交叉验证，生成分割文件，保存为 ${SPLIT_FILE_HOME}/split_%d.json
        为了防止混乱，TXT中保存的是训练和测试对应的UUID，不是df序号
        """
        # 从这里开始 df里的数据顺序不能改变，否则会对不上号
        raise NotImplementedError

    def read_split_file(self, seed: int, ith: int) -> Tuple[List[str]]:
        """ 指定种子和折编号，读取已保存的划分文件
        return train_index, test_index
        """
        raise NotImplementedError

    # 参考：https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
    def read_split_file_innerCV(self, seed: int, outer_cv_ith: int, inner_cv_ith: int) -> Tuple[List[str]]:
        """ 指定种子和折编号，读取已保存的划分文件
        nested cross-validation 比外层少一折
        return train_index, test_index
        """
        raise NotImplementedError

    def save_result(self, data_dict: Dict, seed: int, suffix: str):
        """保存预测结果 以JSON格式保存"""
        filename = os.path.join(self.result_file_dir, 'split_%d_%s.json' % (seed, suffix))
        with open(filename, 'w') as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)

    def read_result(self, seed: int, suffix: str) -> Dict:
        """读取预测结果 以JSON格式保存"""
        filename = os.path.join(self.result_file_dir, 'split_%d_%s.json' % (seed, suffix))
        with open(filename) as f:
            data_dict = json.load(f, encoding='utf-8')
        return data_dict

    def get_result_file_dir(self):
        return self.result_file_dir

    def save_fold_record(self, record: Dict, seed: int, fold: int, suffix: str, float_format: str='%.4f'):
        """ 保存实验记录到文件 ${RESULT_FILE_HOME}/split_<seed>_<suffix>.log,
        同seed同一个suffix的实验结果会追加到同一个文件中
        record 是一个字典，key是str，value必须可以转成str或者是DataFrame
        """
        file_name = os.path.join(self.result_file_dir, 'split_%d_%s.log' % (seed, suffix))

        timestr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        if fold == 0:
            f = open(file_name, mode='w')
        else:
            f = open(file_name, mode='a')
        f.write('\n# ' + '====' * 20 + '\n')
        f.write('@Fold: %d\n@Time: %s\n' % (fold, timestr))
        for key in record:
            f.write('@%s:' % key)
            value = record[key]
            if type(value) == pd.DataFrame:
                f.write('\n')
                value.to_csv(f, sep='\t', float_format=float_format)
            else:
                f.write(str(value))
            f.write('\n')
        f.write('\n')
        f.close()


class KFoldSplitter(DataSplitter):
    """处理关于划分数据集的类"""
    def __init__(self, n_splits: int=10, label_name: str='label',
                sampling_method: str=None,
                split_file_dir: str=None, result_file_dir: str=None):
        """sampling_method: [None, 'down', 'up']
        """
        super(KFoldSplitter, self).__init__(n_splits, split_file_dir, result_file_dir)
        self.label_name = label_name
        self.sampling_method = sampling_method

    def split(self, df: pd.DataFrame, seeds: List[int], group_col_name: str='participant_id'):
        """ 对df做K折交叉验证，生成分割文件，保存为 ${SPLIT_FILE_HOME}/split_%d.json
        为了防止混乱，TXT中保存的是训练和测试对应的UUID，不是df序号
        使用GroupKFold， group信息来源于df_sampled[group_col_name], group_col_name='participant_id'
        """
        # 从这里开始 df里的数据顺序不能改变，否则会对不上号
        print('shape of data_matrix', df.shape)
        self.seeds = seeds
        for seed in seeds:
            self._splitCV(df, seed, group_col_name)

    def _splitCV(self, df: pd.DataFrame, seed: int, group_col_name: str):
        """ 对df做一次K折交叉验证，生成分割文件，保存为 ${SPLIT_FILE_HOME}/split_%d.json
        为了防止混乱，TXT中保存的是训练和测试对应的UUID，不是df序号
        使用GroupKFold， group信息来源于df_sampled[group_col_name], group_col_name='participant_id'
        """
        n_splits = self.n_splits
        if self.sampling_method == 'down':
            df_sampled = self._lower_sampling_data(df)
        elif self.sampling_method == 'up':
            raise NotImplementedError  # TODO: 支持上采样
        elif self.sampling_method is None:
            df_sampled = df
        else:
            raise ValueError(self.sampling_method)

        X = df_sampled.values
        y = df_sampled[self.label_name].values  # .squeeze()

        # kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        groups = df_sampled[group_col_name].values  # 相同人要放在同一折
        kf = RepeatedStratifiedGroupKFold(n_splits=n_splits, random_state=seed)  # shuffle=True, 

        print(kf)

        ith = 0
        file_lines_dict = {}
        for _, test_index in kf.split(X, y=y, groups=groups):  # , groups=groups
            test_index = [df_sampled.index[val] for val in list(test_index)]  # convert numbers to uuid
            file_lines_dict[ith] = ','.join(test_index)
            ith += 1

        filename = self.split_file_dir + 'split_%d.json' % (seed)
        with open(filename, 'w') as f:
            json.dump(file_lines_dict, f, indent=2, ensure_ascii=False)

    def _lower_sampling_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 所有数据存在DataFrame对象df中。数据分为两类：多数类别和少数类别，数据量相差大。
        Label取值为0或1或2(class_num=3)，下采样采取不放回抽样方式
        注意，返回结果没有做打乱，正例都在负例前面
        """
        label_name = self.label_name
        class_num = len(df[label_name].unique())  # 自动计算class_num
        print('balanced (%s %d class).' % (label_name, class_num), end='\t')
        data_list = []
        data_len_list = []
        for class_label in range(class_num):
            data_n = df[df[label_name] == class_label]  # 样本放在data1
            data_n_len = len(data_n)
            data_list.append(data_n)
            data_len_list.append(data_n_len)

        min_len = min(data_len_list)
        lower_data_list = []
        for class_label in range(class_num):
            data_n_len = data_len_list[class_label]
            data_n = data_list[class_label]
            index = np.random.permutation(data_n_len)[:min_len]  # 随机给定下采样样本的序号
            lower_data_n = data_n.iloc[list(index)]  # 下采样
            lower_data_list.append(lower_data_n)
        return pd.concat(lower_data_list)


    def read_split_file(self, seed: int, ith: int) -> Tuple[List[str]]:
        """ 指定种子和折编号，读取已保存的划分文件
        """
        filename = self.split_file_dir + 'split_%d.json' % (seed)
        with open(filename) as f:
            data_dict = json.load(f, encoding='utf-8')
            # lines = [line.strip() for line in f]
        train_index = []
        for key in data_dict:
            if int(key) == ith:
                test_index = data_dict[key].split(',')
            else:
                train_index.extend(data_dict[key].split(','))
        return train_index, test_index

    # 参考：https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
    def read_split_file_innerCV(self, seed: int, outer_cv_ith: int, inner_cv_ith: int) -> Tuple[List[str]]:
        """ 指定种子和折编号，读取已保存的划分文件
        nested cross-validation 比外层少一折
        """
        real_inner_cv_ith = inner_cv_ith
        if inner_cv_ith >= outer_cv_ith:
            real_inner_cv_ith += 1

        filename = self.split_file_dir + 'split_%d.json' % (seed)
        with open(filename) as f:
            data_dict = json.load(f, encoding='utf-8')

        train_index = []
        for key in data_dict:
            if int(key) == outer_cv_ith:
                continue
            elif int(key) == real_inner_cv_ith:
                test_index = data_dict[key].split(',')
            else:
                train_index.extend(data_dict[key].split(','))
        return train_index, test_index


class KFoldSplitterAug(DataSplitter):
    """处理增强后的数据交叉验证划分的类
    必须提供增强前的数据划分结果，生成对应的增强后的数据划分的结果.
    增强后的数据以 <uuid>_xxx 的方式命令，直接判断有没有 '_' 字符就知道是否是增强的数据
    """
    def __init__(self, n_splits: int=10, split_file_dir: str=None, result_file_dir: str=None, test_on_origin: bool=True):
        """sampling_method: [None, 'down', 'up']
        test_on_origin 设为False的话 取出索引和 KFoldSplitter是一样的
        """
        super(KFoldSplitterAug, self).__init__(n_splits, split_file_dir, result_file_dir)
        self.test_on_origin = test_on_origin


    def split(self, df: pd.DataFrame, seeds: List[int], from_split: str, ori_id_col: str='uuid'):
        """ 对df做K折交叉验证，生成分割文件，保存为 ${SPLIT_FILE_HOME}/split_%d.json
        为了防止混乱，TXT中保存的是训练和测试对应的UUID，不是df序号
        df里面至少应该包含clp_id, uuid, label 三列
        """
        # 从这里开始 df里的数据顺序不能改变，否则会对不上号
        print('shape of data_matrix', df.shape)
        k_splitter = KFoldSplitter(split_file_dir=from_split)
        self.seeds = seeds
        for seed in seeds:
            self._splitCV(df, seed, k_splitter, ori_id_col)

    def _splitCV(self, df, seed, k_splitter, ori_id_col):
        """ 读取k_splitter已经划分好的结果, 生成对应的增强后的划分结果
        """
        file_lines_dict = {}
        for ith in range(self.n_splits):
            _, test_ori = k_splitter.read_split_file(seed, ith)
            test_index = df.index[df[ori_id_col].isin(test_ori)]
            file_lines_dict[ith] = ','.join(list(test_index))

        filename = self.split_file_dir + 'split_%d.json' % (seed)
        with open(filename, 'w') as f:
            json.dump(file_lines_dict, f, indent=2, ensure_ascii=False)


    def _is_aug(self, index: str) -> bool:
        # 增强后的数据以 <uuid>_xxx 的方式命令，直接判断有没有 '_' 字符就知道是否是增强的数据
        # return '_' in item
        return len(index) > 6
    
    def read_split_file(self, seed: int, ith: int) -> Tuple[List[str]]:
        """ 指定种子和折编号，读取已保存的划分文件
        """
        filename = self.split_file_dir + 'split_%d.json' % (seed)
        with open(filename) as f:
            data_dict = json.load(f, encoding='utf-8')
            # lines = [line.strip() for line in f]
        train_index = []
        for key in data_dict:
            if int(key) == ith:
                test_index = data_dict[key].split(',')
            else:
                train_index.extend(data_dict[key].split(','))
        if self.test_on_origin:
            # 去掉增强的数据
            test_index = [item for item in test_index if not self._is_aug(item) ]
        return train_index, test_index

    def read_split_file_innerCV(self, seed: int,
                                outer_cv_ith: int, inner_cv_ith: int) -> Tuple[List[str]]:
        """ 指定种子和折编号，读取已保存的划分文件
        nested cross-validation 比外层少一折
        """
        real_inner_cv_ith = inner_cv_ith
        if inner_cv_ith >= outer_cv_ith:
            real_inner_cv_ith += 1

        filename = self.split_file_dir + 'split_%d.json' % (seed)
        with open(filename) as f:
            data_dict = json.load(f, encoding='utf-8')

        train_index = []
        for key in data_dict:
            if int(key) == outer_cv_ith:
                continue
            elif int(key) == real_inner_cv_ith:
                test_index = data_dict[key].split(',')
            else:
                train_index.extend(data_dict[key].split(','))
        if self.test_on_origin:
            # 去掉增强的数据
            test_index = [item for item in test_index if not self._is_aug(item)]
        return train_index, test_index





