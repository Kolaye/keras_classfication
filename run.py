# -*- coding: utf-8 -*-

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import shutil

tf.app.flags.DEFINE_string('mode', 'train', 'optional: train, save_pb, eval')

tf.app.flags.DEFINE_string('multigpu', 'n','optional: y,n')

# params for train
tf.app.flags.DEFINE_string('restore_model_path', '',
                           'a history model you have trained, you can load it and continue trainging')
tf.app.flags.DEFINE_integer('keep_weights_file_num', 10,
                            'the max num of weights files keeps, if set -1, means infinity')
tf.app.flags.DEFINE_integer('num_classes', 54, 'the num of classes which your task should classify')
tf.app.flags.DEFINE_integer('input_size', 224, 'the input image size of the model')
tf.app.flags.DEFINE_integer('batch_size', 8, '')
tf.app.flags.DEFINE_float('learning_rate',0.0001, '')
tf.app.flags.DEFINE_integer('max_epochs', 100, '')


# params for save pb
tf.app.flags.DEFINE_string('freeze_weights_file_path', './save_h5_model',
                           'if it is set, the specified h5 weights file will be converted as a pb model, '
                           'only valid when {mode}=save_pb')

# params for evaluation
tf.app.flags.DEFINE_string('eval_weights_path', '', 'weights file path need to be evaluate')
tf.app.flags.DEFINE_string('eval_pb_path', '', 'a directory which contain pb file needed to be evaluate')


tf.app.flags.DEFINE_string('data_local', '../datasets/train_datasets', 'the train data path on local')
tf.app.flags.DEFINE_string('save_model_local', './save_h5_model', 'the training output results on local')
tf.app.flags.DEFINE_string('test_data_local', '', 'the test data path on local')
tf.app.flags.DEFINE_string('tmp', '', 'a temporary path on local')

FLAGS = tf.app.flags.FLAGS


def check_args(FLAGS):
    if FLAGS.mode not in ['train', 'save_pb', 'eval']:
        raise Exception('FLAGS.mode error, should be train, save_pb or eval')
    if FLAGS.num_classes == 0:
        raise Exception('FLAGS.num_classes error, '
                        'should be a positive number associated with your classification task')

    if FLAGS.mode == 'train':
        if FLAGS.data_local == '':
            raise Exception('you must specify FLAGS.data_local')
        if not os.path.exists(FLAGS.data_local):
            raise Exception('FLAGS.data_local: %s is not exist' % FLAGS.data_local)
        if FLAGS.restore_model_path != '' and (not os.path.exists(FLAGS.restore_model_path)):
            raise Exception('FLAGS.restore_model_path: %s is not exist' % FLAGS.restore_model_path)
        if os.path.isdir(FLAGS.restore_model_path):
            raise Exception('FLAGS.restore_model_path must be a file path, not a directory, %s' % FLAGS.restore_model_path)
        if FLAGS.save_model_local == '':
            raise Exception('you must specify FLAGS.save_model_local')
        elif not os.path.exists(FLAGS.save_model_local):
            os.mkdir(FLAGS.save_model_local)
        if FLAGS.test_data_local != '' and (not os.path.exists(FLAGS.test_data_local)):
            raise Exception('FLAGS.test_data_local: %s is not exist' % FLAGS.test_data_local)

    if FLAGS.mode == 'save_pb':
        if FLAGS.freeze_weights_file_path == '':
            raise Exception('you must specify FLAGS.freeze_weights_file_path when you want to save pb')
        if not os.path.exists(FLAGS.freeze_weights_file_path):
            raise Exception('FLAGS.freeze_weights_file_path: %s is not exist' % FLAGS.freeze_weights_file_path)
        if os.path.isdir(FLAGS.freeze_weights_file_path):
            raise Exception('FLAGS.freeze_weights_file_path must be a file path, not a directory, %s ' % FLAGS.freeze_weights_file_path)
        if os.path.exists(FLAGS.freeze_weights_file_path.rsplit('/', 1)[0] + '/model'):
            raise Exception('a model directory is already exist in ' + FLAGS.freeze_weights_file_path.rsplit('/', 1)[0]
                            + ', please rename or remove the model directory ')

    if FLAGS.mode == 'eval':
        if FLAGS.eval_weights_path == '' and FLAGS.eval_pb_path == '':
            raise Exception('you must specify FLAGS.eval_weights_path '
                            'or FLAGS.eval_pb_path when you want to evaluate a model')
        if FLAGS.eval_weights_path != '' and FLAGS.eval_pb_path != '':
            raise Exception('you must specify only one of FLAGS.eval_weights_path '
                            'and FLAGS.eval_pb_path when you want to evaluate a model')
        if FLAGS.eval_weights_path != '' and (not os.path.exists(FLAGS.eval_weights_path)):
            raise Exception('FLAGS.eval_weights_path: %s is not exist' % FLAGS.eval_weights_path)
        if FLAGS.eval_pb_path != '' and (not os.path.exists(FLAGS.eval_pb_path)):
            raise Exception('FLAGS.eval_pb_path: %s is not exist' % FLAGS.eval_pb_path)
        if not os.path.isdir(FLAGS.eval_pb_path) or (not FLAGS.eval_pb_path.endswith('model')):
            raise Exception('FLAGS.eval_pb_path must be a directory named model '
                            'which contain saved_model.pb and variables, %s' % FLAGS.eval_pb_path)
        if FLAGS.test_data_local == '':
            raise Exception('you must specify FLAGS.test_data_local when you want to evaluate a model')
        if not os.path.exists(FLAGS.test_data_local):
            raise Exception('FLAGS.test_data_local: %s is not exist' % FLAGS.test_data_local)


def main(argv=None):
    check_args(FLAGS)
    if FLAGS.multigpu =='n':

        if FLAGS.mode == 'train':
            from train import train_model
            train_model(FLAGS)
        elif FLAGS.mode == 'save_pb':
            from save_model import load_weights_save_pb
            load_weights_save_pb(FLAGS)
        elif FLAGS.mode == 'eval':
            from eval import eval_model
            eval_model(FLAGS)
    elif FLAGS.multigpu == 'y':
        if FLAGS.mode == 'train':
            from multigpu_train import train_model
            train_model(FLAGS)
        elif FLAGS.mode == 'save_pb':
            from save_multigpu_model import load_weights_save_pb
            load_weights_save_pb(FLAGS)
        elif FLAGS.mode == 'eval':
            from eval import eval_model
            eval_model(FLAGS)
    else:
        raise Exception('Please use true option of multigpu')


if __name__ == '__main__':
    tf.app.run()
