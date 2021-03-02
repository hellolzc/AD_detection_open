
import functools
import tensorflow as tf

# Stop support for tf 1.x
# if tf.__version__ >= '2.0':

from tensorflow import keras
print("[INFO @ %s]"%__name__, "Tensorflow version:", tf.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Concatenate, Flatten, Reshape, Add, Multiply, LeakyReLU, RepeatVector, Permute, Lambda
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, GaussianNoise

from tensorflow.keras.layers import SeparableConv1D, Conv2D, Conv1D
from tensorflow.keras.layers import MaxPooling2D, MaxPooling1D, AveragePooling1D, AveragePooling2D
# from keras.layers import LSTM
from tensorflow.keras.layers import RNN, LSTM, GRU, Bidirectional
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import backend as K


"""
else:
    import keras
    print("[INFO @ %s]"%__name__, "Tensorflow version:", tf.__version__,
        "Keras version:", keras.__version__)
    from keras.models import Sequential
    from keras.layers import Input, Concatenate, Flatten, Reshape
    from keras.layers import Dense, Dropout, Activation, BatchNormalization, GaussianNoise

    from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LeakyReLU, AveragePooling1D, AveragePooling2D
    # from keras.layers import LSTM
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU
    from keras.layers import Bidirectional

    from keras.regularizers import l1, l2
"""

def cnn_0_2d(input_shape):
    """当成一通道的图像，用二维卷积"""
    model = Sequential()
    # default "image_data_format": "channels_last",  input_shape = train_x.shape[1:]
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=[*input_shape, 1]))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3),  activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3),  activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3),  activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3),  activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


def cnn2d_time_wrap(input_shape):
    """沿时间轴弯折成K通道的图像，用二维卷积，网络类似VGG16"""
    model = Sequential()
    #  f_len = input_shape[0]
    #  f_wid = input_shape[1]
    model.add(Reshape((64, 64, -1), input_shape=input_shape))
    model.add(GaussianNoise(0.05))
    # default "image_data_format": "channels_last"

    # , kernel_regularizer=l2(0.00001)  # , use_bias=False
    reg_factor = 0.0001
    for filter_num in [64, 128]:
        model.add(Conv2D(filter_num, (3,3), strides=1, padding='same', kernel_regularizer=l2(reg_factor)))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Conv2D(filter_num, (3,3), strides=1, padding='same', kernel_regularizer=l2(reg_factor)))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2)))

    for filter_num in [256, 256]:
        model.add(Conv2D(filter_num, (3,3), strides=1, padding='same', kernel_regularizer=l2(reg_factor)))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Conv2D(filter_num, (3,3), strides=1, padding='same', kernel_regularizer=l2(reg_factor)))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Conv2D(filter_num, (3,3), strides=1, padding='same', kernel_regularizer=l2(reg_factor)))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(reg_factor)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(reg_factor)))
    return model


def cnn_1d(input_shape):
    """全部使用一维卷积"""
    model = Sequential()
    # default "image_data_format": "channels_last"

    model.add(Conv1D(128, 3, strides=2, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.5))

    for filter_num in [128, 128, 128]:
        model.add(Conv1D(filter_num, 3, strides=2, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

def cnn_1d_bn_IS18(input_shape):
    """InterSpeech 2018 GCNN作者所用"""
    model = Sequential()
    # default "image_data_format": "channels_last"

    model.add(Conv1D(64, 3, strides=1, input_shape=input_shape, padding='same',
                            use_bias=False, kernel_initializer='random_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.5))

    for filter_num in [64, 64, 64, 64, 64]:
        model.add(Conv1D(filter_num, 3, strides=1, padding='same',
                                use_bias=False, kernel_initializer='random_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, use_bias=False, kernel_initializer='random_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def cnn_1d_bn_tuned(input_shape):
    model = Sequential()
    # default "image_data_format": "channels_last"

    model.add(Conv1D(32, 8, strides=1, input_shape=input_shape, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(4))
    model.add(Dropout(0.5))

    for filter_num in [32, 64, 64, 128, 128]:
        model.add(Conv1D(filter_num, 4, strides=1, padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(4))
        model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


def cnn1d_lstm(input_shape):
    model = Sequential()
    model.add(Conv1D(32, 8, strides=2, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    for filter_num in [64, 64]:
        model.add(Conv1D(filter_num, 8, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(48, return_sequences=True), merge_mode='concat'))  # returns a sequence of vectors  , dropout=0.25
    model.add(Dropout(0.5))  # Attention
    model.add(Bidirectional(LSTM(48), merge_mode='concat'))  # return a single vector  , dropout=0.25
    model.add(Dense(1, activation='sigmoid'))
    return model


def cnn2d_cnn1d_lstm(input_shape):
    ''' CNN-LSTM like Emnet'''
    reg_factor = 0.00001
    inp = Input(shape=input_shape)
    
    x = Reshape((*input_shape, 1))(inp)
    # x = GaussianNoise(0.05)(x)
    x = Conv2D(64, (6,1), strides=1, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)

    x = Reshape((x.shape[-3], x.shape[-2] * x.shape[-1]))(x)

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
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(48, return_sequences=True), merge_mode='concat')(x)  # returns a sequence of vectors  , dropout=0.25 , recurrent_dropout
    x = Dropout(0.5)(x)  # TODO: Try Attention
    x = Bidirectional(LSTM(48, return_sequences=True), merge_mode='concat')(x)  # return a single vector  , dropout=0.25

    x = MaxPooling1D(pool_size=x.shape[-2])(x)
    # x2 = AveragePooling1D(pool_size=512)(x)
    # x = Concatenate()([x1, x2])
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(inputs=inp, outputs=out)
    return model


def EmNet_sequential(input_shape):
    """modified emnet"""
    model = Sequential()
    # default "image_data_format": "channels_last"
    # assert K.image_data_format() == 'channels_last':

    reg_factor = 0.00001  #   , kernel_regularizer=l1(0.00001) , kernel_regularizer=l1(0.0002)
    model.add(Reshape((*input_shape, 1), input_shape=input_shape))
    # model.add(GaussianNoise(0.05))
    model.add(Conv2D(64, (6,1), strides=1, padding='same'))  # , kernel_regularizer=l1(0.001)
    # model.add(BatchNormalization())
    model.add(Activation('relu'))  # model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(4, 1)))

    model.add(Reshape((input_shape[0]//2, 64*input_shape[1])))
    
    model.add(Conv1D(128, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    
    model.add(Conv1D(128, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(256, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(48, return_sequences=True), merge_mode='concat'))  # returns a sequence of vectors
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(48, return_sequences=True), merge_mode='concat'))  # return a single vector, dropout=0.25
    model.add(MaxPooling1D(pool_size=128))
    
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


def classifier_wavnet(shape_):
    raise NotImplementedError
    
    def cbr(x, out_layer, kernel, stride, dilation):
        x = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x
    
    def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2**i for i in range(n)]
        x = Conv1D(filters = filters,
                   kernel_size = 1,
                   padding = 'same')(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same', 
                              activation = 'tanh', 
                              dilation_rate = dilation_rate)(x)
            sigm_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same',
                              activation = 'sigmoid', 
                              dilation_rate = dilation_rate)(x)
            x = Multiply()([tanh_out, sigm_out])
            x = Conv1D(filters = filters,
                       kernel_size = 1,
                       padding = 'same')(x)
            res_x = Add()([res_x, x])
        return res_x
    
    inp = Input(shape = (shape_))
    x = cbr(inp, 64, 7, 1, 1)
    # x = BatchNormalization()(x)
    x = wave_block(x, 16, 3, 12)
    x = BatchNormalization()(x)
    x = wave_block(x, 32, 3, 8)
    x = BatchNormalization()(x)
    x = wave_block(x, 64, 3, 4)
    x = BatchNormalization()(x)
    x = wave_block(x, 128, 3, 1)
    x = cbr(x, 32, 7, 1, 1)
    # x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    out = Dense(11, activation = 'softmax', name = 'out')(x)
    
    model = models.Model(inputs = inp, outputs = out)
    
    opt = Adam(lr = LR)
    opt = tfa.optimizers.SWA(opt)
    model.compile(loss = losses.CategoricalCrossentropy(), optimizer = opt, metrics = ['accuracy'])
    return model


def sequence_attention_pooling(input_activations, attention_vec_name='attention_vec'):
    """ 输入激活维度为 （Batchsize, Timestep, Dimension）
        Attention 后输出维度为 （Batchsize, Dimension）
    """
    # compute importance for each step
    units = int(input_activations.shape[-1])
    attention = Dense(1, activation='tanh', use_bias=False)(input_activations)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(units)(attention)
    attention = Permute([2, 1], name=attention_vec_name)(attention)

    sent_representation = Multiply()([input_activations, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(units,))(sent_representation)
    return sent_representation

class AttentionBlock(keras.layers.Layer):
    """Attention Pooling
    Input Shape: (Batchsize, Timestep, Dim)
    Output Shape: (Batchsize, Dim)
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
        self.repeat_1 = RepeatVector(self._units)
        self.permute_1 = Permute([2, 1], name=self.attention_vec_name)
        self.merge_mul = Multiply()
        self.lambda_1 = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(self._units,))

    def call(self, inputs):
        att = self.activation_1(self.flatten_1(self.linear_1(inputs)))
        att = self.permute_1(self.repeat_1(att))
        x = self.merge_mul([inputs, att])
        x = self.lambda_1(x)
        return x

def cnn2d_cnn1d_lstm_attention(input_shape):
    ''' CNN-LSTM like Emnet'''
    reg_factor = 0.00001
    inp = Input(shape=input_shape)
    
    x = Reshape((*input_shape, 1))(inp)
    # x = GaussianNoise(0.05)(x)
    x = Conv2D(64, (6,1), strides=1, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)

    x = Reshape((x.shape[-3], x.shape[-2] * x.shape[-1]))(x)

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
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(48, return_sequences=True), merge_mode='concat')(x)  # returns a sequence of vectors  , dropout=0.25 , recurrent_dropout
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(48, return_sequences=True), merge_mode='concat')(x)  # return a single vector  , dropout=0.25

    # x = MaxPooling1D(pool_size=x.shape[-2])(x)
    x = sequence_attention_pooling(x)
    # x2 = AveragePooling1D(pool_size=512)(x)
    # x = Concatenate()([x1, x2])
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(inputs=inp, outputs=out)
    return model

def cnn2d_cnn1d_lstm_attention_concat_cnn_multihead(input_shape):
    ''' CNN-LSTM like Emnet'''
    reg_factor = 0.00001
    inp = Input(shape=input_shape)
    
    x = Reshape((*input_shape, 1))(inp)
    # x = GaussianNoise(0.05)(x)
    x = Conv2D(64, (6,1), strides=1, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)

    x = Reshape((x.shape[-3], x.shape[-2] * x.shape[-1]))(x)

    x = Conv1D(128, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor))(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(128, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor))(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(256, 3, strides=1, padding='same', kernel_regularizer=l2(reg_factor))(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    x_root = BatchNormalization()(x)

    # Path 1
    x = Dropout(0.5)(x_root)
    x = Bidirectional(LSTM(48, return_sequences=True), merge_mode='concat')(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(48, return_sequences=True), merge_mode='concat')(x)
    x = sequence_attention_pooling(x, attention_vec_name='attention_vec_lstm')
    x = Flatten()(x)
    x1 = Dense(32, activation='relu')(x)

    # Path 2
    # att_list = []
    # for i in range(3):
    #     att_list.append(sequence_attention_pooling(x_root, attention_vec_name='attention_vec%d' % (i+1)))

    # x = Concatenate()(att_list)
    x = sequence_attention_pooling(x_root)
    x = Flatten()(x)
    x2 = Dense(32, activation='relu')(x)


    # Concate
    x = Concatenate()([x1, x2])
    x = Dropout(0.5)(x)

    out = Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(inputs=inp, outputs=out)
    return model

def cnn2d_cnn1d_attention(input_shape):
    ''' CNN-Attention'''
    reg_factor = 0.00001
    inp = Input(shape=input_shape)
    
    x = Reshape((*input_shape, 1))(inp)
    # x = GaussianNoise(0.05)(x)
    x = Conv2D(64, (6,1), strides=1, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)

    x = Reshape((x.shape[-3], x.shape[-2] * x.shape[-1]))(x)

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

    x = Concatenate()(att_list)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(inputs=inp, outputs=out)
    return model

def get_model_creator(model_choose: str):
    """ 返回一个函数: 函数参数是 input_shape, 返回值是一个 keras.model"""
    if model_choose == 'cnn_0_2d':
        return cnn_0_2d
    elif model_choose == 'cnn_1d':
        return cnn_1d
    elif model_choose == 'cnn_1d_bn_tuned':
        return cnn_1d_bn_tuned
    elif model_choose == 'cnn1d_lstm':
        return cnn1d_lstm
    elif model_choose == 'cnn2d_cnn1d_lstm':
        return cnn2d_cnn1d_lstm
    elif model_choose == 'EmNet_sequential':
        return EmNet_sequential
    elif model_choose == 'cnn2d_cnn1d_attention':
        return cnn2d_cnn1d_attention
    elif model_choose == 'cnn2d_cnn1d_lstm_attention':
        return cnn2d_cnn1d_lstm_attention
    elif model_choose == 'cnn2d_cnn1d_lstm_attention_concat_cnn_multihead':
        return cnn2d_cnn1d_lstm_attention_concat_cnn_multihead
    else:
        raise ValueError('Not Supported Model %s' % model_choose)
