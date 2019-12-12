# -*- coding: utf-8 -*-
import os
import shutil
import codecs
import numpy as np
from glob import glob

from PIL import Image
import tensorflow as tf
from keras import backend
from keras.optimizers import adam

from tensorflow.python.saved_model import tag_constants

from train import model_fn
from save_model import load_weights
import json
backend.set_image_data_format('channels_last')


# 实现保存之后的模型的评估
# 对于保存后的pb模型的读取并评估测试模型的准确率


with open('./label.json','r') as f:
    label_id_names = json.load(f)

def center_img(img, size=None, fill_value=255):
    """
    center img in a square background
    """
    h, w = img.shape[:2]
    if size is None:
        size = max(h, w)
    shape = (size, size) + img.shape[2:]  #元组的链接，变成三元组或者四元组
    background = np.full(shape, fill_value, np.uint8)
    center_x = (size - w) // 2
    center_y = (size - h) // 2
    background[center_y:center_y + h, center_x:center_x + w] = img
    return background


def preprocess_img(img_path, img_size):
    """
    image preprocessing
    you can add your special preprocess mothod here
    """
    img = Image.open(img_path)
    ##保证同比缩放
    resize_scale = img_size / max(img.size[:2])  
    img = img.resize((int(img.size[0] * resize_scale), int(img.size[1] * resize_scale)))

    img = img.convert('RGB')
    img = np.array(img)
    img = img[:, :, ::-1]  #转换为BGR
    img = center_img(img, img_size)
    return img


def load_test_data(FLAGS):

    label_files = glob(os.path.join(FLAGS.test_data_local, '*.txt'))
    test_data = np.ndarray((len(label_files), FLAGS.input_size, FLAGS.input_size, 3),
                           dtype=np.uint8)
    img_names = []
    test_labels = []
    for index, file_path in enumerate(label_files):
        with codecs.open(file_path, 'r', 'utf-8') as f:
            line = f.readline()
        line_split = line.strip().split(', ')
        if len(line_split) != 2:
            print('%s contain error lable' % os.path.basename(file_path))
            continue
        try:
            test_data[index] = preprocess_img(os.path.join(FLAGS.test_data_local, line_split[0]), FLAGS.input_size)
            img_names.append(line_split[0])
            test_labels.append(int(line_split[1]))
        except:
            img_names.append(line_split[0])
            test_labels.append(int(line_split[1]))
            continue
    return img_names, test_data, test_labels


def test_single_model(FLAGS):

    pb_model_dir = FLAGS.eval_pb_path

    signature_key = 'predict_image'
    input_key_1 = 'input_img'
    output_key_1 = 'output_score'
    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.get_default_graph().as_default():

        sess1 = tf.Session(graph=tf.Graph(), config=config)
        meta_graph_def = tf.saved_model.loader.load(sess1, [tag_constants.SERVING], pb_model_dir)
        # print(meta_graph_def)

        signature = meta_graph_def.signature_def
        # print(signature)
        input_images_tensor_name = signature[signature_key].inputs[input_key_1].name
        # print(input_images_tensor_name)
        output_score_tensor_name = signature[signature_key].outputs[output_key_1].name
        # print(output_score_tensor_name)

        input_images = sess1.graph.get_tensor_by_name(input_images_tensor_name)
        output_score = sess1.graph.get_tensor_by_name(output_score_tensor_name)


    img_names, test_data, test_labels = load_test_data(FLAGS)
    print(len(test_labels))
    print(test_data.shape)
    print(len(img_names))
    right_count = 0
    error_infos = []

    from collections import defaultdict
    true_label_number = defaultdict(int)
    predict_true_number = defaultdict(int)

    for index, img in enumerate(test_data):
        img = img[np.newaxis, :, :, :]  # the input tensor shape of resnet is [?, 224, 224, 3]
        img = img.astype(np.float32) / 225
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        img[..., 0] -= mean[0]
        img[..., 1] -= mean[1]
        img[..., 2] -= mean[2]
        img[..., 0] /= std[0]
        img[..., 1] /= std[1]
        img[..., 2] /= std[2]
        pred_score = sess1.run([output_score], feed_dict={input_images: img})
        if pred_score is not None:
            pred_label = np.argmax(pred_score[0], axis=1)[0]
            try:
                test_label = test_labels[index]
                with open('./re_label.txt', 'a')as f1:
                    f1.write(str(test_label))
                    f1.write('\n')
                with open('./pr_label.txt', 'a')as f2:
                    f2.write(str(pred_label))
                    f2.write('\n')                    

                true_label_number[str(test_label)] +=1
                # print(test_label)
                if pred_label == test_label:
                    predict_true_number[str(test_label)] +=1
                    right_count += 1
                else:
                    error_infos.append('%s, %s, %s\n' % (img_names[index], test_label, pred_label))
            except:
                break
        else:
            print('pred_score is None')
    accuracy = right_count / len(img_names)
    print('accuracy: %s' % accuracy)
    for k in true_label_number.keys():
        print(label_id_names[str(k)] + ' accuracy: %s\n' %(predict_true_number[k]/true_label_number[k]))
    result_file_name = os.path.join(FLAGS.eval_pb_path, 'accuracy1.txt')
    with open(result_file_name, 'w') as f:
        f.write('# predict error files\n')
        f.write('####################################\n')
        f.write('file_name, true_label, pred_label\n')
        f.writelines(error_infos)
        f.write('####################################\n')

        for k in true_label_number.keys():
            f.write(label_id_names[str(k)]  + ' accuracy: %s\n' %(predict_true_number[k]/true_label_number[k]))
        
        f.write('accuracy: %s\n' % accuracy)
    print('end')


def eval_model(FLAGS):
    if FLAGS.eval_weights_path != '':
        if os.path.isdir(FLAGS.eval_weights_path):
            test_batch_h5(FLAGS)
        else:
            test_single_h5(FLAGS, FLAGS.eval_weights_path)
    elif FLAGS.eval_pb_path != '':
        test_single_model(FLAGS)


