# -*- coding:utf-8 -*-

"""
将以文件夹分类的图像数据集转换为，label数据存放在与图像同名的txt文件中
"""

import shutil
import os
import json

original_path = '../dir_datasets/'

#大类的类别名称
main_labels = os.listdir(original_path)


target_path = './datasets'

with open('./label.json','r') as f:
    
    label_id_name_dict  = json.load(f)
# print(label_id_name_dict)
# print(label_id_name_dict[str(0)])

label_id = {}
for k,v in label_id_name_dict.items():
    label_id[v] = k

# print(label_id)

i = 0
for main_label in main_labels:

    #数据集分为大类与小类,main_label为大类的类别名称
    path = os.path.join(original_path, main_label)

    #小类的类别名称
    label_names = os.listdir(path)

    for label_name in label_names:

        img_path = os.path.join(path, label_name)
        img_lists = os.listdir(img_path)

        complete_label_name = main_label + '/' + label_name

        for img in img_lists:

            init_path = os.path.join(img_path, img)

            shutil.copy(init_path, target_path)
            new_name = 'limg' + '_' + str(i) + '.' +img.split('.')[-1]
            i += 1
            os.rename(target_path+'/'+img, target_path+'/'+new_name)

            with open(target_path+'/'+ new_name.split('.')[0] + '.txt', 'w') as f:
                text = new_name + ', ' + label_id[complete_label_name]
                f.write(text)



