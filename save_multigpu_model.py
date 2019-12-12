# -*- coding: utf-8 -*-
import os

import tensorflow as tf
from keras import backend
from keras.optimizers import adam
import shutil


from multigpu_train import model_fn


def load_weights(model, weighs_file_path):

    if os.path.isfile(weighs_file_path):
        print('load weights from %s' % weighs_file_path)
        model.load_weights(weighs_file_path)
        print('load weights success')

    else:
        print('load weights failed! Please check weighs_file_path')


def save_pb_model(FLAGS, model):
    if FLAGS.mode == 'train':
        pb_save_dir_local = FLAGS.save_model_local

    elif FLAGS.mode == 'save_pb':
        freeze_weights_file_dir = FLAGS.freeze_weights_file_path.rsplit('/', 1)[0]
        pb_save_dir_local = freeze_weights_file_dir


    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'input_img': model.get_input_at(0)}, outputs={'output_score': model.get_output_at(0)})

    builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(pb_save_dir_local, 'model'))

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    builder.add_meta_graph_and_variables(
        sess=backend.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_image': signature,
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:signature
        },
        legacy_init_op=legacy_init_op)

    builder.save()
    shutil.copytree(pb_save_dir_local + '/model', './deployment/model')
    print('save pb to local path success')



def load_weights_save_pb(FLAGS):

    optimizer = adam(lr=FLAGS.learning_rate, clipnorm=0.001)
    objective = 'categorical_crossentropy'
    metrics = ['accuracy']
    model,p_model= model_fn(FLAGS, objective, optimizer, metrics)
    load_weights(p_model, FLAGS.freeze_weights_file_path)
    print(model.get_input_at(0))
    save_pb_model(FLAGS, model)
