import math
import numpy as np
import matplotlib.pyplot as plt
import inspect, html
import tensorflow as tf
from functools import partial

if tf.__version__ >= '2.0':
    from tensorflow import keras
    print("[INFO @ %s]"%__name__, "Tensorflow version:", tf.__version__)
    from tensorflow.keras.models import Sequential

    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.utils import plot_model
    from tensorflow.keras.utils import multi_gpu_model
    from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

    from tensorflow.keras import backend as K

else:
    import keras
    print("[INFO @ %s]"%__name__, "Tensorflow version:", tf.__version__,
        "Keras version:", keras.__version__)
    from keras.models import Sequential

    from keras.utils import to_categorical
    from keras.utils import plot_model
    from keras.utils import multi_gpu_model
    from keras.callbacks import LearningRateScheduler, EarlyStopping

    from keras import backend as K


from ad_detection.mlcode.model_base_class import Model

def shuffle_train_data(X_train, Y_train):
    # 只打乱训练集
    shuffle_index = np.random.permutation(X_train.shape[0])
    X_train = X_train[shuffle_index]
    Y_train = Y_train[shuffle_index]
    return X_train, Y_train, shuffle_index

def generate_arrays_from_data(X_train, Y_train, sample_size):
    # e.g.
    # X shape: (13368, 1000, 130) (1547, 1000, 130)
    # Y shape: (13368,) (1547,)
    steps_per_epoch = int(np.ceil(X_train.shape[0]/sample_size))
    while True:
        # 每个epoch做一次shuffle
        X_train, Y_train, _ = shuffle_train_data(X_train, Y_train)
        for j in range(steps_per_epoch):  # [0,1,...,steps_per_epoch-1]
            start_indx = j*sample_size
            end_indx = (j+1)*sample_size
            if end_indx > X_train.shape[0]:
                end_indx = X_train.shape[0]
            X_j = X_train[start_indx:end_indx, :]
            Y_j = Y_train[start_indx:end_indx]
            yield (X_j, Y_j)


def narrow_exp_decay(epoch, lr_init=0.001, decay_step=10, start_decay=0, lr_final=1.0e-5):
    """narrow expontional decay"""
    if epoch < start_decay:
        return lr_init
    drop = 0.5
    lrate = lr_init * math.pow(drop, float(epoch-start_decay) / decay_step)
    return max(lrate, lr_final)

# def step_decay(epoch, lr_init=0.001, decay_step=10):
#     drop = 0.9  # 0.5
#     lrate = lr_init * (drop ** (epoch // decay_step))
#     return lrate

def keras_decay(step_no, lr_init, decay_rate):
    # not used in call back
    return lr_init * 1 / (1 + decay_rate * step_no)

class KerasModelAdapter(Model):
    """将keras模型装饰一下，从而将所有的模型设置集中到一处"""
    def __init__(self, input_shape=None, model_creator=None, **params):
        """model_creator是一个函数，输入为input_shape, 返回一个Keras Model"""
        self.params = params.copy()  # 存档
        self.lr = params.pop('lr', 0.001)
        self.lr_decay_method = params.pop('lr_decay_method', 'keras')
        if self.lr_decay_method == 'keras':
            self.decay_rate = params.pop('decay_rate', 0.0)
        else:
            assert self.lr_decay_method == 'exponential'
            self.decay_step = params.pop('decay_step', 10)
        self.epochs = params.pop('epochs', 300)
        self.verbose = params.pop('verbose', 0)
        self.gpus = params.pop('gpus', 1)
        self.batch_size = params.pop('batch_size', 64 * self.gpus)
        self.loss = params.pop('loss', 'binary_crossentropy')

        if 'name' not in params:
            params['name'] = 'KerasModelAdapter'
        super().__init__(params)  # 用不上的参数传给父类

        K.clear_session()  # 清理掉旧模型
        self.input_shape = input_shape
        self.model_creator = model_creator
        model = model_creator(input_shape)
        self.model = model
        self.out_dim = model.output_shape[1]
        self.train_history = None

    def set_hyper_params(self, **hyper_params):
        # TODO: finish this method
        raise NotImplementedError


    def __str__(self):
        return self.summary()

    def __repr__(self):
        try:
            func_src = inspect.getsource(self.model_creator)
            func_src = html.unescape(func_src)
        except IOError:
            func_src = "nocode"
        return repr({'code':func_src, 'params':self.params, 'input_shape':self.input_shape})

    def _load_model(self, to_load):
        """
        Load the model weights from the given path.

        Args:
            to_load (str): path to the saved model file in h5 format.

        """
        try:
            self.model.load_weights(to_load)
        except:
            raise Exception("Invalid saved file provided")


    def save_model(self, save_path):
        """
        Save the model weights to `save_path` provided.
        """
        self.model.save_weights(save_path)

    def summary(self):
        stringlist = []
        stringlist.append('Params:' + str(self.params))
        self.model.summary(print_fn=lambda x: stringlist.append(x), line_length=90)
        short_model_summary = "\n".join(stringlist)
        return short_model_summary

    def plot_model(self, file_path='./log/model2.png', show_shapes=True):
        plot_model(self.model, to_file=file_path, show_shapes=show_shapes)

    def _report_lr(self, steps_per_epoch):
        if self.lr_decay_method == 'keras':
            print('lr = lr_init * 1 / (1 + decay_rate * step_no)')
            if self.decay_rate != 0.0:
                new_lr = self.lr * 1 / (1 + self.decay_rate*steps_per_epoch)
                print("LearningRate will be %f after 1 epoch." % new_lr)
                new_lr = self.lr * 1 / (1 + self.decay_rate*steps_per_epoch*self.epochs)
                print("LearningRate will be %f after last epoch." % new_lr)
        else:
            print('lr = lr_init * (drop ** (epoch / decay_step))')
            new_lr = narrow_exp_decay(self.epochs//2, self.lr, self.decay_step)
            print("LearningRate will be %f after %d epoch." % (new_lr, self.epochs//2))
            new_lr = narrow_exp_decay(self.epochs, self.lr, self.decay_step)
            print("LearningRate will be %f after last epoch." % new_lr)

    def _compile_model(self):
        """fit之前需要先compile"""
        if self.gpus > 1:
            self.model = multi_gpu_model(self.model, self.gpus)
        if self.lr_decay_method == 'keras':
            opt = keras.optimizers.Adam(lr=self.lr, decay=self.decay_rate)
        else:
            opt = keras.optimizers.Adam(lr=self.lr)
        self.model.compile(optimizer=opt, loss=self.loss, metrics=['accuracy'])

    def _get_callback(self):
        callbacks=None
        if self.lr_decay_method == 'exponential':
            callbacks = []
            lrate = LearningRateScheduler(
                partial(narrow_exp_decay, lr_init=self.lr, decay_step=self.decay_step))
            callbacks.append(lrate)
            # es = EarlyStopping(monitor='val_loss', patience=5)
        return callbacks

    def fit(self, X_train, Y_train, validation_data=None):
        """return None"""
        # history = _model.fit(train_x, train_y, epochs=10, batch_size=32, validation_data=(val_x, val_y))
        # return self.model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=validation_data)
        #
        if self.out_dim >= 2:
            Y_train = to_categorical(Y_train)
        if validation_data is not None:
            val_x, val_y = validation_data
            if self.out_dim >= 2:
                val_y = to_categorical(val_y)
            validation_data = (val_x, val_y)
        batch_size = self.batch_size
        my_generator = generate_arrays_from_data(X_train, Y_train, batch_size)
        steps_per_epoch = int(np.ceil(X_train.shape[0]/batch_size))
        print("[INFO @ %s]"%__name__, "SampleNum:", X_train.shape[0], 'StepsPerEpoch:', steps_per_epoch,
            'Batchsize:', batch_size)
        self._report_lr(steps_per_epoch)

        self._compile_model()
        callbacks = self._get_callback()

        self.train_history = self.model.fit(my_generator,  # _generator
                                    epochs=self.epochs, verbose=self.verbose,
                                    steps_per_epoch=steps_per_epoch,
                                    validation_data=validation_data,
                                    callbacks=callbacks)
        self.trained = True
        self.show_history()


    def fit_generator(self, train_generator, val_generator,
            train_set_size, val_set_size):
        """return None
        TODO: support out_dim >=2
        """
        # history = _model.fit(train_x, train_y, epochs=10, batch_size=32, validation_data=(val_x, val_y))
        # return self.model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=validation_data)
        #
        assert self.out_dim == 1
        dataiter = iter(val_generator)
        val_x, val_y = next(dataiter)

        validation_data = (val_x, val_y)
        batch_size = self.batch_size

        sample_num = train_set_size
        steps_per_epoch = int(np.ceil(sample_num/batch_size))
        print("[INFO @ %s]"%__name__, "SampleNum:", sample_num, 'StepsPerEpoch:', steps_per_epoch,
            'Batchsize:', batch_size)
        self._report_lr(steps_per_epoch)

        self._compile_model()
        callbacks = self._get_callback()

        self.train_history = self.model.fit(train_generator,  # _generator
                                    epochs=self.epochs, verbose=self.verbose,
                                    steps_per_epoch=steps_per_epoch,
                                    validation_data=validation_data,
                                    validation_steps=5,
                                    use_multiprocessing=False,
                                    workers=3,
                                    callbacks=callbacks)
        self.trained = True
        self.show_history()


    def predict(self, X):
        if self.out_dim >= 2:  # softmax
            return np.squeeze(np.argmax(self.model.predict(X).squeeze(), axis=1))
        else:  # sigmoid
            return np.squeeze(np.round(self.model.predict(X)))

    def predict_generator(self, val_generator, val_set_size=None, val_batch_size=None):
        assert self.out_dim == 1
        result_list = []
        if val_set_size is None:
            steps_per_epoch = 1
        else:
            steps_per_epoch = int(np.ceil(val_set_size/val_batch_size))

        dataiter = iter(val_generator)

        for i in range(steps_per_epoch):
            batch_x, batch_y = next(dataiter)
            # print("[INFO @ %s]"%__name__, f"Predict {len(batch_y)} samples")
            batch_result =  np.squeeze(np.round(self.model.predict(batch_x)))
            result_list.append(batch_result)
        return np.concatenate(result_list, axis=-1)

    def predict_proba(self, X):
        if self.gpus > 1:
            # avoid error "CUDNN_STATUS_BAD_PARAM" when using 2 gpu
            ori_length = len(X)
            padded_shape = list(X.shape)
            padded_shape[0] = int(np.ceil(ori_length / float(self.batch_size)) * self.batch_size)
            X_padded = np.zeros(padded_shape, dtype=X.dtype)
            X_padded[0:ori_length] = X
            return self.model.predict(X_padded)[0:ori_length]
        return self.model.predict(X)

    def clone_model(self):
        """reset graph and return a deep copy of this model object"""
        # 载入新模型，会自动清理内存，意味着旧模型报废了
        new_model = KerasModelAdapter(input_shape=self.input_shape, model_creator=self.model_creator, **self.params)
        return new_model

    def show_history(self):
        """将训练过程可视化的函数"""
        history = self.train_history
        print(history.history.keys())
        fig = plt.figure(figsize=(15,4))

        ax = plt.subplot(121)
        plt.plot(history.history['accuracy'])  # 'acc'
        plt.plot(history.history['val_accuracy'])
        ax.set_ylim([0.2, 1.0])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower left')

        ax = plt.subplot(122)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        # ax.set_ylim([0.0, 3.0])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()
