
import functools
import numpy as np
import tensorflow as tf

# Stop support for tf 1.x
# if tf.__version__ >= '2.0':

from tensorflow import keras
print("[INFO @ %s]"%__name__, "Tensorflow version:", tf.__version__)

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization

from tensorflow.keras.layers import Conv2D, Conv1D
from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D
# from keras.layers import LSTM
from tensorflow.keras.layers import RNN, LSTM, GRU, Bidirectional
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import backend as K

from ad_detection.dlcode.util.position import AddPositionalEncoding
from ad_detection.dlcode.util.transformer import TransformerBlock

def sequence_attention_pooling(input_activations, attention_vec_name='attention_vec'):
    """ 输入激活维度为 （Batchsize, Timestep, Dimension）
        Attention 后输出维度为 （Batchsize, Dimension）
    """
    # compute importance for each step
    units = int(input_activations.shape[-1])
    attention = Dense(1, activation='tanh', use_bias=False)(input_activations)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = layers.RepeatVector(units)(attention)
    attention = layers.Permute([2, 1], name=attention_vec_name)(attention)

    sent_representation = layers.Multiply()([input_activations, attention])
    sent_representation = layers.Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(units,))(sent_representation)
    return sent_representation

class AttentionBlock(keras.layers.Layer):
    """Attention Pooling
    Input Shape: (Batchsize, Timestep, Dim)
    Output Shape: (Batchsize, Dim)
    att = softmax(tanh(w*x))
    y = sum(att * x_i)
    See: https://keras.io/guides/making_new_layers_and_models_via_subclassing/
    """
    def __init__(self, attention_vec_name='attention_vec'):
        super(AttentionBlock, self).__init__()
        self._units = None
        self.attention_vec_name = attention_vec_name

    def build(self, input_shape):
        self._units = int(input_shape[-1])
        self.linear_1 = Dense(1, activation='tanh', use_bias=False)
        self.flatten_1 = Flatten()
        self.activation_1 = Activation('softmax')
        self.repeat_1 = layers.RepeatVector(self._units)
        self.permute_1 = layers.Permute([2, 1], name=self.attention_vec_name)
        self.merge_mul = layers.Multiply()
        self.lambda_1 = layers.Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(self._units,))

    def call(self, inputs):
        att = self.activation_1(self.flatten_1(self.linear_1(inputs)))
        att = self.permute_1(self.repeat_1(att))
        x = self.merge_mul([inputs, att])
        x = self.lambda_1(x)
        return x

class ScaledAttentionBlock(keras.layers.Layer):
    """Scaled Attention Pooling
    Input Shape: (Batchsize, Timestep, Dim)
    Output Shape: (Batchsize, Dim)
    att = softmax(tanh(w*x))
    y = sum(att * x_i)
    See: https://keras.io/guides/making_new_layers_and_models_via_subclassing/
    """
    def __init__(self, dim_K, attention_vec_name='attention_vec'):
        super(ScaledAttentionBlock, self).__init__()
        # self._units = None  # channel
        self._dim = dim_K  # d_model // num_heads
        self.attention_vec_name = attention_vec_name

    def build(self, input_shape):
        # self._units = int(input_shape[-1])
        # self.scale_q = Dense(self._dim, use_bias=False)
        self.scale_v = Dense(self._dim, use_bias=False)
        
        #
        self.linear_1 = Dense(1, activation='tanh', use_bias=False)
        self.flatten_1 = Flatten()
        self.activation_1 = Activation('softmax')
        self.repeat_1 = RepeatVector(self._dim)
        self.permute_1 = Permute([2, 1], name=self.attention_vec_name)
        self.merge_mul = Multiply()
        self.lambda_1 = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(self._dim,))

    def call(self, inputs):
        q = inputs  # self.scale_q(inputs)
        v = self.scale_v(inputs)
        # sqrt_d = K.constant(np.sqrt(self._dim), dtype=K.floatx())
        # att = self.activation_1(self.flatten_1(self.linear_1(q) / sqrt_d))
        att = self.activation_1(self.flatten_1(self.linear_1(q)))
        att = self.permute_1(self.repeat_1(att))
        x = self.merge_mul([v, att])
        x = self.lambda_1(x)
        return x

####################### CNN-LSTM ####################################


def fc_cnn1d_lstm_attention_ablation(input_shape):
    ''' CNN-LSTM like Emnet'''
    reg_factor = 0.00001
    inp = Input(shape=input_shape)

    x = Dense(256)(inp)  # , kernel_regularizer=l2(reg_factor)
    x = Activation("relu")(x)

    # x = Conv1D(128, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor))(x)
    # x = Activation('relu')(x)
    # x = MaxPooling1D(2)(x)

    # x = Conv1D(256, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor))(x)
    # x = Activation('relu')(x)
    # x = MaxPooling1D(2)(x)

    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    # x = Bidirectional(LSTM(48, return_sequences=True), merge_mode='concat')(x)  # returns a sequence of vectors  , dropout=0.25 , recurrent_dropout
    # x = Dropout(0.5)(x)
    # x = Bidirectional(LSTM(48, return_sequences=True), merge_mode='concat')(x)  # return a single vector  , dropout=0.25

    # x = MaxPooling1D(pool_size=x.shape[-2])(x)
    x = sequence_attention_pooling(x)
    # x = AveragePooling1D(pool_size=x.shape[-2])(x)
    # x = layers.Concatenate()([x1, x2])
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(inputs=inp, outputs=out)
    return model


def fc_cnn1d_lstm_attention(input_shape):
    ''' CNN-LSTM like Emnet'''
    reg_factor = 0.00001
    inp = Input(shape=input_shape)

    x = Dense(256)(inp)  # , kernel_regularizer=l2(reg_factor)
    x = Activation("relu")(x)

    x = Conv1D(128, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor))(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(256, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor))(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    # # resnet block
    # y = Conv1D(256, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor))(x)
    # y = Activation('relu')(y)
    # y = Conv1D(256, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor))(y)
    # x = layers.Add()([x, y])
    # x = Activation('relu')(x)

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(48, return_sequences=True), merge_mode='concat')(x)  # returns a sequence of vectors  , dropout=0.25 , recurrent_dropout
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(48, return_sequences=True), merge_mode='concat')(x)  # return a single vector  , dropout=0.25

    # x = MaxPooling1D(pool_size=x.shape[-2])(x)
    x = sequence_attention_pooling(x)
    # x2 = AveragePooling1D(pool_size=512)(x)
    # x = layers.Concatenate()([x1, x2])
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(inputs=inp, outputs=out)
    return model



def fc_cnn1d_attention(input_shape):
    ''' CNN - MultiHead Attention'''
    reg_factor = 0.00001
    inp = Input(shape=input_shape)

    x = Dense(256)(inp)
    x = Activation("relu")(x)

    x = Conv1D(128, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor))(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(128, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor))(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(256, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor))(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    att_list = []
    for i in range(3):
        att_list.append(sequence_attention_pooling(x, attention_vec_name='attention_vec%d' % (i+1)))

    x = layers.Concatenate()(att_list)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(inputs=inp, outputs=out)
    return model


##################### Positional Encoding ###########################

def fc_cnn1d_pe_attention(input_shape):
    ''' CNN - Positional Encoding - Attention'''
    reg_factor = 0.00001
    inp = Input(shape=input_shape)

    x = Dense(256, kernel_regularizer=l2(reg_factor))(inp)
    x = Activation("relu")(x)

    x = Conv1D(128, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor))(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(128, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor))(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(256, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor))(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    x = BatchNormalization()(x)
    x = AddPositionalEncoding()(x)
    # x = Dropout(0.5)(x)
    att_list = []
    for i in range(3):
        # att_list.append(sequence_attention_pooling(x, attention_vec_name='attention_vec%d' % (i+1)))
        att_list.append(AttentionBlock(attention_vec_name='attention_vec%d' % (i+1))(x))

    x = layers.Concatenate()(att_list)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(inputs=inp, outputs=out)
    return model




def sequence_multihead_attention_pooling(input_activations, head_num=8, d_model=512):
    """ 输入激活维度为 （Batchsize, Timestep, Dimension）
        Attention 后输出维度为 （Batchsize, d_model）
    """
    x = input_activations
    att_list = []
    for i in range(head_num):
        att_list.append(ScaledAttentionBlock(d_model // head_num,
                        attention_vec_name='attention_vec%d' % (i+1))(x))

    x = layers.Concatenate()(att_list)
    # sent_representation = Dense(d_model, use_bias=False)(x)
    return x  # sent_representation


def fc_cnn1d_pe_scaledattention(input_shape):
    ''' CNN - Positional Encoding - scaled attention'''
    reg_factor = 0.00001
    inp = Input(shape=input_shape)

    x = Dense(128)(inp)
    x = Activation("relu")(x)

    x = Conv1D(128, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor))(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(128, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor))(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(256, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor))(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    x = BatchNormalization()(x)
    x = AddPositionalEncoding()(x)
    # x = Dropout(0.5)(x)
    x = sequence_multihead_attention_pooling(x, 4, 512)

    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(inputs=inp, outputs=out)
    return model

