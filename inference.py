# -*- coding:utf-8 -*-
# @time :2019.09.17
# @IDE : pycharm
# @autor :lxztju
# @github : https://github.com/lxztju


##对保存后的模型进行单张图像的推理预测，输出为预测的类别的名称

import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

pb_path = "./save_h5_model/model"

signature_key = 'predict_image'

with open('./label.json','r') as f:
    label_id_names = json.load(f)

img = Image.open('../train_data/img_1.jpg')
#img = np.array(img)
input_size=380


def center_img(img, size=None, fill_value=255):
    """
    center img in a square background
    """
    h, w = img.shape[:2]
    if size is None:
        size = max(h, w)
    shape = (size, size) + img.shape[2:]
    background = np.full(shape, fill_value, np.uint8)
    center_x = (size - w) // 2
    center_y = (size - h) // 2
    background[center_y:center_y + h, center_x:center_x + w] = img
    return background

def preprocess_img(img):
    """
    image preprocessing
    you can add your special preprocess method here
    """
    resize_scale = input_size / max(img.size[:2])
    img = img.resize((int(img.size[0] * resize_scale), int(img.size[1] * resize_scale)))
    img = img.convert('RGB')
    img = np.array(img)
    img = img[:, :, ::-1]
    img = center_img(img,input_size)
    return img

img = preprocess_img(img)

img = img[np.newaxis, :, :, :]  # the input tensor shape of resnet is [?, 224, 224, 3]
img = img.astype(np.float32) / 225
mean = [0.5236, 0.5897, 0.6573]
std = [0.3918, 0.3476, 0.3666]

img[..., 0] -= mean[0]
img[..., 1] -= mean[1]
img[..., 2] -= mean[2]
img[..., 0] /= std[0]
img[..., 1] /= std[1]
img[..., 2] /= std[2]

config = tf.ConfigProto(allow_soft_placement=True)
with tf.get_default_graph().as_default():
    sess = tf.Session(graph=tf.Graph(), config=config)
    # load modelc

    meta_graph_def = tf.saved_model.loader.load(sess, [tag_constants.SERVING], pb_path)

    # get signature
    signature = meta_graph_def.signature_def

    # get tensor name
    in_tensor_name = signature[signature_key].inputs['input_img'].name
    out_tensor_name = signature[signature_key].outputs['output_score'].name

    # get tensor
    input_images = sess.graph.get_tensor_by_name(in_tensor_name)
    output_score = sess.graph.get_tensor_by_name(out_tensor_name)

    # run
    pred_score = sess.run([output_score], feed_dict={input_images: img})
    pred_label = np.argmax(pred_score[0], axis=1)[0]
    print(label_id_names[str(pred_label)])
    sess.close()
