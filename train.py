# -*- coding: utf-8 -*-
import multiprocessing
import os
import shutil
from glob import glob
import tensorflow as tf
import numpy as np
from keras import backend
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard, Callback
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam, RMSprop

from data_gen_label import data_flow
from warmup_cosine_decay_scheduler import WarmUpCosineDecayScheduler
import efficientnet.keras as efn

from bilinear_pooling import bilinear_pooling

backend.set_image_data_format('channels_last')


def model_fn(FLAGS, objective, optimizer, metrics):

    # base_model = efn.EfficientNetB3(include_top=False,
    #                                 input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
    #                                 classes=FLAGS.num_classes, )
    # # input_size =  380                      
    model = efn.EfficientNetB4(include_top=False,
                                    input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                                    classes=FLAGS.num_classes, )
    # # input_size =  456     
    # base_model = efn.EfficientNetB5(include_top=False,
    #                                 input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
    #                                 classes=FLAGS.num_classes, )

    # # input_size =  528    
    # base_model = efn.EfficientNetB6(include_top=False,
    #                                 input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
    #                                 classes=FLAGS.num_classes, )

    # for i, layer in enumerate(model.layers):
    #     if "batch_normalization" in layer.name:
    #         model.layers[i] = GroupNormalization(groups=32, axis=-1, epsilon=0.00001)
    x = model.output

    # 插入双线性池化操作
    # x = bilinear_pooling(x）

    # x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.4)(x)
    predictions = Dense(FLAGS.num_classes, activation='softmax')(x)  # activation="linear",activation='softmax'
    model = Model(input=model.input, output=predictions)

    model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    model.summary( )
    return model


class Mycbk(Callback):
    def __init__(self, FLAGS, model):
        super(Mycbk, self).__init__( )
        self.FLAGS = FLAGS

    def on_epoch_end(self, epoch, logs={}):
        save_path = os.path.join(self.FLAGS.save_model_local, 'weights_%03d.h5' % (epoch))

        self.model.save_weights(save_path)
        print('save weights file', save_path)

        if self.FLAGS.keep_weights_file_num > -1:
            weights_files = glob(os.path.join(self.FLAGS.save_model_local, '*.h5'))
            if len(weights_files) >= self.FLAGS.keep_weights_file_num:
                weights_files.sort(key=lambda file_name: os.stat(file_name).st_ctime, reverse=True)
                os.remove(weights_files[-1])


def train_model(FLAGS):
    preprocess_input = efn.preprocess_input

    train_sequence, validation_sequence = data_flow(FLAGS.data_local, FLAGS.batch_size,
                                                    FLAGS.num_classes, FLAGS.input_size, preprocess_input)

    optimizer = Adam(lr=FLAGS.learning_rate)

    objective = 'categorical_crossentropy'
    metrics = ['accuracy']

    model = model_fn(FLAGS, objective, optimizer, metrics)

    if FLAGS.restore_model_path != '' and os.path.exists(FLAGS.restore_model_path):
        model.load_weights(FLAGS.restore_model_path)
        print("LOAD OK!!!")

    if not os.path.exists(FLAGS.save_model_local):
        os.makedirs(FLAGS.save_model_local)

    log_local = './log_file/'

    tensorBoard = TensorBoard(log_dir=log_local)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, mode='auto')

    sample_count = len(train_sequence) * FLAGS.batch_size
    epochs = FLAGS.max_epochs
    warmup_epoch = 5
    batch_size = FLAGS.batch_size
    learning_rate_base = FLAGS.learning_rate
    total_steps = int(epochs * sample_count / batch_size)
    warmup_steps = int(warmup_epoch * sample_count / batch_size)

    warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                            total_steps=total_steps,
                                            warmup_learning_rate=0,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=0,
                                            )

    cbk = Mycbk(FLAGS)
    model.fit_generator(
        train_sequence,
        steps_per_epoch=len(train_sequence),
        epochs=FLAGS.max_epochs,
        verbose=1,
        callbacks=[cbk, tensorBoard, warm_up_lr],
        validation_data=validation_sequence,
        max_queue_size=10,
        workers=int(multiprocessing.cpu_count( ) * 0.7),
        use_multiprocessing=True,
        shuffle=True
    )

    print('training done!')

    from save_model import save_pb_model
    save_pb_model(FLAGS, model)

    if FLAGS.test_data_local != '':
        print('test dataset predicting...')
        from eval import load_test_data
        img_names, test_data, test_labels = load_test_data(FLAGS)
        test_data = preprocess_input(test_data)
        predictions = model.predict(test_data, verbose=0)

        right_count = 0
        for index, pred in enumerate(predictions):
            predict_label = np.argmax(pred, axis=0)
            test_label = test_labels[index]
            if predict_label == test_label:
                right_count += 1
        accuracy = right_count / len(img_names)
        print('accuracy: %0.4f' % accuracy)
        metric_file_name = os.path.join(FLAGS.save_model_local, 'metric.json')
        metric_file_content = '{"total_metric": {"total_metric_values": {"accuracy": %0.4f}}}' % accuracy
        with open(metric_file_name, "w") as f:
            f.write(metric_file_content + '\n')
    print('end')
