#!/usr/bin/env python
# -*- coding: utf-8 -*-
# hellolzc 20190924
from typing import List, Tuple, Dict
import sys
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ad_detection.mlcode.data_manager import DataSet
from ad_detection.dlcode.util.data_augmentation import mask_policy

from tensorflow import keras

DEBUG = False
def set_debug_print(debug_print):
    """set DEBUG
    """
    global DEBUG
    DEBUG = debug_print


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, list_IDs: List[str], data_file_path:str, labels: Dict,
                batch_size: int=32, data_shape: Tuple[int]=(2560,512), 
                scaler: StandardScaler=None,
                shuffle: bool=True,
                mask_aug_num: int=0):
        """Initialization"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.data_file_path = data_file_path

        self.batch_size = batch_size
        self.data_shape = data_shape
        self.scaler = scaler
        # self.n_classes = n_classes
        self.shuffle = shuffle
        self.mask_aug_num = mask_aug_num

        self.feat_clps = h5py.File(self.data_file_path, "r", rdcc_nbytes=1024*1024*2048)

        self.on_epoch_end()

    def __del__(self):
        self.feat_clps.close()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.data_shape))
        # with h5py.File(self.data_file_path, "r") as feat_clps:
        X = DLDataSet.get_X_scaled(self.data_shape, self.feat_clps, list_IDs_temp,
                scaler=self.scaler, mask_aug_num=self.mask_aug_num)

        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        # y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y


class DLDataSet(DataSet):
    """ Manager DeepLearning Dataset
    self.data_file_path : The path of the file to store hdf5 dataset.
    self.label_se : A pandas series to store label.
    """
    def __init__(self, hdf5_file_path, label_file_path, class_num,
            normlize=True, train_batch_size=32, mask_aug_num=0):
        """ hdf5_file_path: 对应的hdf5文件提供每个sample的特征矩阵
            label_file_path: csv文件，第一列必须为index_col, 必须有名为label列, 
                            且索引和标签与hdf5文件中数据一一对应
        """
        super(DLDataSet, self).__init__()
        self.data_file_path = hdf5_file_path  # 数据文件的位置
        print('HDF5 File: %s' % hdf5_file_path)
        label_df = pd.read_csv(label_file_path, encoding='utf-8', index_col=0)
        self.label_df = label_df
        self.label_se = label_df['label'].copy()  # 标签信息
        self.data_length = None  # 一个utterace的长度
        self.class_num = class_num
        self.normlize = normlize
        self.mask_aug_num = mask_aug_num
        self.train_batch_size = train_batch_size
        self.scaler_dict = {}

    def set_data_length(self, max_length):
        """设置截断或补齐的长度位置"""
        self.data_length = max_length

    def summary(self, single_str=False):
        """返回描述字符串list"""
        summary_list = []
        summary_list.append('Data File: %s' % str(self.data_file_path))
        summary_list.append('Data Length: %s' % str(self.data_length))
        summary_list.append('Train batch size: %s' % str(self.train_batch_size))
        summary_list.append('Class num: %s' % str(self.class_num))
        summary_list.append('Normlize: %s' % str(self.normlize))
        summary_list.append('Mask Augmentation: %s' % str(self.mask_aug_num))
        if single_str:
            return '\n'.join(summary_list)
        else:
            return summary_list

    def describe_data(self):
        """用来检查hdf5文件中存放的数据的shape信息
        返回一个numpy array, 包含所有数据的shape
        """
        with h5py.File(self.data_file_path, "r") as feat_clps:
            shape_list = []
            for key in feat_clps:
                data_shape_i = feat_clps[key].shape  # np.array(feat_clps[key]).shape
                shape_list.append(data_shape_i)
            print('data num:', len(shape_list))
            shape_mat = np.array(shape_list)
            for dim_no in range(shape_mat.shape[1]):
                col_i = shape_mat[:,dim_no]
                print('Dim %d mean:%d, min:%d, max:%d, std:%f' % \
                    (dim_no, col_i.mean(), col_i.min(), col_i.max(), col_i.std()))
            return shape_mat

    def get_clp_id_uuid_label_df(self):
        """write for CascadePipelineCV 针对增强的数据
        返回的dataframe有['uuid','label','participant_id'(可选)] 三列, 索引列名为 clp_id
        """
        label_df = self.label_df[['uuid', 'participant_id', 'label']]
        assert label_df.index.name == 'clp_id'
        return label_df

    def get_input_shape(self):
        """返回要训练的数据样本维度信息"""
        with h5py.File(self.data_file_path, "r") as feat_clps:
            # firstkey = list(feat_clps.keys())[0]
            for key in feat_clps:
                firstkey = key
                break
            firstdata_shape = feat_clps[firstkey].shape # np.array(feat_clps[firstkey]).shape
            print('[INFO @ %s]'%__name__, 'First data shape:', firstdata_shape, end='\t')

        data_shape = list(firstdata_shape)
        if self.data_length is None:
            print('Use First data shape')
            return data_shape
        else:
            data_shape[0] = self.data_length
            print('Length is set to %d, so shape is' % self.data_length, data_shape)
            return data_shape

    @staticmethod
    def _fit_X_scaler(data_shape, feat_clps, index_clp_id):
        """处理X的截断和标准化问题
        计算X的均值和标准差以便normlize, 返回StandardScaler对象，
        """
        max_length = data_shape[0]
        # 截断
        sc = StandardScaler()
        for _, clp_id in enumerate(index_clp_id):
            data_i_truncated = feat_clps[clp_id][:max_length, :]
            # 拟合数据集X_train，返回存下来拟合参数
            sc.partial_fit(data_i_truncated)
        return sc

    @staticmethod
    def get_X_scaled(data_shape, feat_clps, index_clp_id, scaler=None, mask_aug_num=None):
        """处理X的截断和标准化问题
        使用计算好的X的均值和标准差进行normlize
        scaler = None 则不标准化
        """
        max_length = data_shape[0]
        # 标准化之后进行 补0或截断
        # X_train = np.row_stack([np.array(feat_clps[clp_id])[None] for clp_id in train_index_clp_id])
        X_train = np.zeros([len(index_clp_id), *data_shape])
        for indx, clp_id in enumerate(index_clp_id):
            data_i_truncated = feat_clps[clp_id][:max_length, :]  # 截断到最大长度
            actual_length = data_i_truncated.shape[0]
            if scaler is not None:  # normlize
                data_i_truncated = scaler.transform(data_i_truncated)
            if mask_aug_num is not None:
                data_i_truncated = mask_policy(data_i_truncated, mask_prob=1.0, num=mask_aug_num)

            if actual_length < max_length:
                X_train[indx, :actual_length, :] = data_i_truncated  # 补0到最大长度
            else:
                X_train[indx, :, :] = data_i_truncated
        return X_train

    @staticmethod
    def get_Y(clps_df, index_clp_id):
        """类似_get_X_scaled的method
        """
        return clps_df.loc[index_clp_id].values.squeeze()

    def get_data_scaled(self, seed, ith, normlize=None, data_splitter=None):
        """ 将X,Y划分成训练集和测试集并在训练集上做标准化，将参数应用到测试集上
        可能会消耗过多内存，建议用下面的生成器替代
        eg.
            X_train, X_test, Y_train, Y_test = get_data_scaled(1998, 2)
        description:
            X_train.shape = (n_samples, x_len, x_dim)
            X_test.shape = (n_samples, )
        """
        data_splitter = data_splitter or self.data_splitter
        if data_splitter is None:
            raise Exception("Provide a data_splitter to split data\n")
        if normlize is None:
            normlize = self.normlize

        clps_df = self.label_se
        data_shape = self.get_input_shape()  # Note: data_shape[0] == self.data_length

        train_index, test_index = data_splitter.read_split_file(seed, ith)
        # print("TRAIN:", train_index, "TEST:", test_index)

        # 从文件取数据
        with h5py.File(self.data_file_path, "r") as feat_clps:
            if normlize:
                sc = self._fit_X_scaler(data_shape, feat_clps, train_index)
            else:
                sc = None
            X_train = self.get_X_scaled(data_shape, feat_clps, train_index, scaler=sc)
            X_test = self.get_X_scaled(data_shape, feat_clps, test_index, scaler=sc)
            Y_train = clps_df.loc[train_index].values.squeeze()
            Y_test = clps_df.loc[test_index].values.squeeze()

        # print('X shape:', X_train.shape, X_test.shape)
        # print('Y shape:', Y_train.shape, Y_test.shape)
        info_dict = {
            'train_index': train_index,
            'test_index': test_index,
        }

        return X_train, X_test, Y_train, Y_test, info_dict


    @staticmethod
    def shuffle_index(X_index: List):
        """打乱训练集, X_index可以是List TODO:check it"""
        # 只打乱训练集
        X_index = np.array(X_index)
        shuffle_index = np.random.permutation(len(X_index))
        X_index = X_index[shuffle_index]
        return X_index, shuffle_index

    def generate_arrays_from_data(self, train_indx: List[str],
                                batchsize: int, data_shape: Tuple[int],
                                scaler: StandardScaler,
                                shuffle: bool=True):
        # e.g.
        # X shape: (13368, 1000, 130) (1547, 1000, 130)
        # Y shape: (13368,) (1547,)
        train_indx_len = len(train_indx)
        steps_per_epoch = int(np.ceil(train_indx_len / batchsize))
        while True:
            # 每个epoch做一次shuffle
            if shuffle:
                train_indx, _ = self.shuffle_index(train_indx)
            for j in range(steps_per_epoch):  # [0,1,...,steps_per_epoch-1]
                start_indx = j * batchsize
                end_indx = (j+1) * batchsize
                if end_indx > train_indx_len:
                    end_indx = train_indx_len
                train_indx_j = train_indx[start_indx:end_indx]
                # read and normalize data
                with h5py.File(self.data_file_path, "r") as feat_clps:
                    X_j = self.get_X_scaled(data_shape, feat_clps, train_indx_j, scaler=scaler, mask_aug_num=self.mask_aug_num)
                Y_j = self.get_Y(self.label_se, train_indx_j)
                yield (X_j, Y_j)

    def generate_train_data(self, train_indx: List[str],
                            batchsize: int, data_shape: Tuple[int],
                            scaler: StandardScaler,
                            shuffle: bool=True):
        if shuffle:
            # 这个generator支持多进程 实测多 tensorflow 2.3 该方法支持不好
            return DataGenerator(train_indx, data_file_path=self.data_file_path, labels=self.label_se.to_dict(),
                batch_size=batchsize, data_shape=data_shape, scaler=scaler, shuffle=True, mask_aug_num=self.mask_aug_num)
        else:
            # 这个generator可以遍历所有数据, 但是最后一个batch可能不是完整的
            return self.generate_arrays_from_data(train_indx, batchsize, data_shape, scaler, shuffle=False)


    def generate_test_data(self, test_indx: List[str],
                            data_shape:Tuple[int],
                            scaler:StandardScaler):
        """ only for test loader
        testloader每次都返回整个测试集
        """
        while True:
            with h5py.File(self.data_file_path, "r") as feat_clps:
                X_test = self.get_X_scaled(data_shape, feat_clps, test_indx, scaler=scaler)
            Y_test = self.get_Y(self.label_se, test_indx)
            yield (X_test, Y_test)

    def get_data_loader(self, seed, ith, normlize=None, data_splitter=None, shuffle=True):
        """ 将X,Y划分成训练集和测试集并在训练集上做标准化，将参数应用到测试集上
        返回生成器以节约内存
        eg.
            trainloader, testloader, info_dict = get_data_scaled(1998, 2)
        description:
            # trainloader每次都返回一个batch
            for X_train, Y_train in trainloader:
                X_train.shape == (batchsize, x_len, x_dim)
                Y_train.shape == (batchsize, )
            # testloader每次都返回整个测试集
            dataiter = iter(testloader)
            X_test, Y_test = dataiter.next()
        """
        data_splitter = data_splitter or self.data_splitter
        if data_splitter is None:
            raise Exception("Provide a data_splitter to split data\n")
        if normlize is None:
            normlize = self.normlize
        train_batch_size = self.train_batch_size

        clps_df = self.label_se
        data_shape = self.get_input_shape()  # Note: data_shape[0] == self.data_length

        train_index, test_index = data_splitter.read_split_file(seed, ith)
        # print("TRAIN:", train_index, "TEST:", test_index)

        # 计算均值方差
        if normlize:
            sc_key = '%d_%d' % (seed, ith)
            if sc_key in self.scaler_dict:
                sc = self.scaler_dict[sc_key]
            else:
                with h5py.File(self.data_file_path, "r") as feat_clps:
                    sc = self._fit_X_scaler(data_shape, feat_clps, train_index)
                    self.scaler_dict[sc_key] = sc
        else:
            sc = None
        # 返回生成器
        trainloader = self.generate_train_data(train_index, train_batch_size, data_shape, sc, shuffle=shuffle)
        testloader = self.generate_test_data(test_index, data_shape, sc)

        info_dict = {
            'train_index': train_index,
            'test_index': test_index,
            'train_size': len(train_index),
            'test_size' : len(test_index),
            'train_batch_size': train_batch_size,
            'test_batch_size' : len(test_index),
            'Y_train': self.get_Y(self.label_se, train_index),
            'Y_test': self.get_Y(self.label_se, test_index),
        }
        return trainloader, testloader, info_dict