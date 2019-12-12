# -*- coding:utf-8 -*-

from matplotlib import pyplot as plt
import os
import pandas as pd
import matplotlib.font_manager as fmgr
YaHei = fmgr.FontProperties(fname='/home/lxztju/anaconda3/chinese.msyh.ttf')

## 数据集类别数据的数量统计，并进行柱状图可视化显示

path = '../dir_datasets_original'

main_labels = os.listdir(path)

main_labels = [os.path.join(path,main_label) for main_label in main_labels]


img_nums = []
label_names = []
img_label_dict = {}
for  label_path in main_labels:

    sub_labels = os.listdir(label_path)

    for sub_label in sub_labels:

        sub_label_path = os.path.join(label_path, sub_label)

        img_num = len(os.listdir(sub_label_path))
        img_nums.append(img_num)
        label_names.append(sub_label)
        img_label_dict[sub_label] = img_num

# for k,v in img_label_dict.items():
#     if v < 60:
#         print(k)
#         with open('name1.txt', 'a')as f:
#             f.write(k)
#             f.write('\n')

label = list(img_label_dict.keys())
print(type(label))
plt.figure(1)
plt.figure(figsize=(20,10))

# ax1 = plt.subplot(2,1,1)
# ax2 = plt.subplot(2,1,2)
# plt.sca(ax1)
# plt.bar(range(len(img_nums)), img_nums,color='rgb',tick_label=label_names)
# plt.sca(ax2)
plt.bar(img_label_dict.keys(), img_label_dict.values(), color='rgb') 
plt.xticks(list(img_label_dict.keys()),img_label_dict.keys(), rotation=90, fontsize=1,fontproperties=YaHei)
plt.savefig('img_num_ori.jpg',dpi=500)
# plt.show()
