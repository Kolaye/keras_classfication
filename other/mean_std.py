import os
from PIL import Image
import numpy as np
 
## 图像数据集的均值与方差的计算

root_path = '../train_data'
_filename = os.listdir(root_path)
filename = []
for _file in _filename:
    if not _file.endswith('.txt'):
        filename.append(_file)

#均值之和
R_channel_m = 0
G_channel_m = 0
B_channel_m = 0

#方差之和
R_channel_s = 0
G_channel_s = 0
B_channel_s = 0
num = len(filename)
for i in range(len(filename)):

    img = Image.open(os.path.join(root_path, filename[i]))
    img = img.convert('RGB')
    img = np.array(img)
    img = img[:, :, ::-1]  #转换为BGR
    img = img.astype(np.float32) / 225

    B_channel_m = B_channel_m + np.sum(img[:, :, 0])/(img.shape[0]* img.shape[1])
    G_channel_m = G_channel_m + np.sum(img[:, :, 1])/(img.shape[0]* img.shape[1])
    R_channel_m = R_channel_m + np.sum(img[:, :, 2])/(img.shape[0]* img.shape[1])
 
B_mean = B_channel_m / num
G_mean = G_channel_m / num
R_mean = R_channel_m / num


for i in range(len(filename)):

    img = Image.open(os.path.join(root_path, filename[i]))
    img = img.convert('RGB')
    img = np.array(img)
    img = img[:, :, ::-1]
    img = img.astype(np.float32) / 225

    B_channel_s = B_channel_s + np.sum(np.power(img[:, :, 0]-R_mean, 2) )/(img.shape[0]* img.shape[1])
    G_channel_s = G_channel_s + np.sum(np.power(img[:, :, 1]-G_mean, 2) )/(img.shape[0]* img.shape[1])
    R_channel_s = R_channel_s + np.sum(np.power(img[:, :, 2]-B_mean, 2) )/(img.shape[0]* img.shape[1])

B_std = np.sqrt(B_channel_s/num)
G_std = np.sqrt(G_channel_s/num)
R_std = np.sqrt(R_channel_s/num)
 
with open('mean_std.txt','w')as f:
    text = "B_mean is %f, G_mean is %f, R_mean is %f" % (B_mean, G_mean, R_mean) + '\n' + "B_std is %f, G_std is %f, R_std is %f" % (B_std, G_std, R_std)  
    f.write(text)
print("B_mean is %f, G_mean is %f, R_mean is %f" % (B_mean, G_mean, R_mean))
print("B_std is %f, G_std is %f, R_std is %f" % (B_std, G_std, R_std))
