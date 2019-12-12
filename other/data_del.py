# -*- coding:utf-8 -*-

import os
import numpy as np
from PIL import Image

#将警告变为异常可以捕捉
import warnings
warnings.filterwarnings("error", category=UserWarning)
##由于网上爬虫爬的图，有好多是错误的会有EXIF警告，需要处理，
#还有一些图像打不开因此需要处理

###### 删除一些破损的图像，以防在训练模型中报错


path = '../datasets'
filelists = os.listdir(path)

input_size=300


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

for _file in filelists:
    if _file.endswith('.txt'):
        pass
    else:
        try:
            img_path = os.path.join(path,_file)
            img = Image.open(img_path)
            img = preprocess_img(img)
        except:
            os.remove(img_path)
            os.remove(os.path.join(path, _file.split('.')[0] + '.txt'))
            print(img_path)
