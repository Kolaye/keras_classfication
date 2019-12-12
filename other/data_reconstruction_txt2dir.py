# -*- coding:utf-8 -*-

"""
将包含类别信息的txt数据集类型，转换为文件夹形式
"""

import shutil
import os
import json

original_path = './datasets/'

target_path = './dir_datasets'

with open('./label.json','r') as f:
    
    label_id_name_dict  = json.load(f)



for j in range(len(label_id_name_dict)):
    if not os.path.exists(os.path.join(target_path,label_id_name_dict[str(j)])):
        os.makedirs(os.path.join(target_path,label_id_name_dict[str(j)]))


file_list = os.listdir(original_path)
# print(file_list)

for i in range(len(file_list)):

    if file_list[i].endswith('.txt'):
        txt_path = os.path.join(original_path,file_list[i])

        with open(txt_path,'r') as f:
            line = f.readlines()
            # print(line[0].split(' ')[-1])
            label = line[0].strip().split(' ')[-1]
            name = line[0].split(',')[0]

            img_path = os.path.join(original_path, name)

        dst = os.path.join(target_path,label_id_name_dict[label])

        shutil.copy(img_path,dst)
    # break
