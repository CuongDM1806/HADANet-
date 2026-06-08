import mne
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
import numpy as np
from model import *
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os

from wulaoshi.cho_model_1 import DomainAdaptationModel

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DomainAdaptationModel().to(device)

model.load_state_dict(torch.load("D:\pythonProject/svr/wulaoshi/physionet_29_brain_model.pth"))
# model = model.to(torch.device("cpu"))
# 测试集
test_features = np.load("D:\postgraduate\MIdata\physionet\libo\physionet\DE/S29.npy")
test_labels = np.load("D:\postgraduate\MIdata\physionet\libo\physionet\label/S29.npy")
test_features = np.array(test_features)
test_labels = np.array(test_labels)
test_features = test_features.reshape(-1, 90, 4, 64, 5)
test_features = np.transpose(test_features, (0, 1, 3, 4, 2))
test_features = test_features.reshape((-1, 64, 5, 4))
test_labels = test_labels.reshape((-1, 90, 4, 1)).mean(axis=2)
test_labels = test_labels.reshape(-1)
# print(test_features.shape)
# print(test_labels.shape)
# 定义5个区域
# test_features_partition1 = []  # 1-3,33-37
# test_features_partition2 = []  # 4-11,38-47
# test_features_partition3 = []  # 12-15,48-52
# test_features_partition4 = []  # 16-24,31-32,53-61
# test_features_partition5 = []  # 25-30,62-64

# test_features_partition1_1 = test_features[:, 0:3, :, :]
# test_features_partition1_2 = test_features[:, 32:37, :, :]
# test_features_partition1 = np.concatenate((test_features_partition1_1, test_features_partition1_2), axis=1)
# # print(features_partition1.shape)  # (8100, 8, 5, 4)
#
# test_features_partition2_1 = test_features[:, 3:11, :, :]
# test_features_partition2_2 = test_features[:, 37:47, :, :]
# test_features_partition2 = np.concatenate((test_features_partition2_1, test_features_partition2_2), axis=1)
# # # print(features_partition2.shape)  # (8100, 18, 5, 4)
# #
# test_features_partition3_1 = test_features[:, 11:15, :, :]
# test_features_partition3_2 = test_features[:, 47:52, :, :]
# test_features_partition3 = np.concatenate((test_features_partition3_1, test_features_partition3_2), axis=1)
# # # print(features_partition3.shape)  # (8100, 9, 5, 4)
# #
# test_features_partition4_1 = test_features[:, 15:24, :, :]
# test_features_partition4_2 = test_features[:, 30:32, :, :]
# test_features_partition4_3 = test_features[:, 52:61, :, :]
# test_features_partition4 = np.concatenate((test_features_partition4_1, test_features_partition4_2, test_features_partition4_3), axis=1)
# # # print(features_partition4.shape)  # (8100, 20, 5, 4)
# #
# test_features_partition5_1 = test_features[:, 24:30, :, :]
# test_features_partition5_2 = test_features[:, 61:64, :, :]
# test_features_partition5 = np.concatenate((test_features_partition5_1, test_features_partition5_2), axis=1)# 测试集
# test_features_partition1 = torch.tensor(test_features_partition1, dtype=torch.float32)
# test_features_partition2 = torch.tensor(test_features_partition2, dtype=torch.float32)
# test_features_partition3 = torch.tensor(test_features_partition3, dtype=torch.float32)
# test_features_partition4 = torch.tensor(test_features_partition4, dtype=torch.float32)
# test_features_partition5 = torch.tensor(test_features_partition5, dtype=torch.float32)
test_features = torch.tensor(test_features, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# testData = TensorDataset(test_features_partition1, test_features_partition2, test_features_partition3, test_features_partition4,
#                             test_features_partition5, test_labels)
#
# test_dataloader = DataLoader(testData, batch_size=3394, shuffle=True, drop_last=True)

# x1, y = data
# print(test_features_partition1.shape)torch.Size([90, 8, 5, 4])
# print(test_labels.shape)torch.Size([90])
# x1 = test_features_partition1.to(device)
# x2 = test_features_partition2.to(device)
# x3 = test_features_partition3.to(device)
# x4 = test_features_partition4.to(device)
# x5 = test_features_partition5.to(device)
x = test_features.to(device)
y = test_labels.to(device)
# print(test_labels.shape)  # 左手：23 右手：22 双手：21 双脚：24
# print(test_labels)
outputs, brain = model(x)

# print(output32[:].shape)

output64 = brain.to(torch.device("cpu"))
# output64 = np.array(output64)
# 读取导联位置信息，创建对应的info

# 读取MNE中biosemi电极位置信息
# biosemi_montage = mne.channels.make_standard_montage('biosemi64')
# print(biosemi_montage.get_positions())
# sensor_data = biosemi_montage.get_positions()['ch_pos']
# print(sensor_data)
# sensor_dataframe = pd.DataFrame(sensor_data).T
# print(sensor_dataframe)
# sensor_dataframe.to_excel('sensor_dataframe.xlsx')

# 获取的除ch_pos外的信息
'''
'coord_frame': 'unknown', 'nasion': array([ 5.27205792e-18,  8.60992398e-02, -4.01487349e-02]),
'lpa': array([-0.08609924, -0.        , -0.04014873]), 'rpa': array([ 0.08609924,  0.        , -0.04014873]),
'hsp': None, 'hpi': None
'''

# 将获取的电极位置信息修改并补充缺失的电极位置，整合为1020.xlsx
data1020 = pd.read_excel('D:\pythonProject\svr/brainPic/physionet_chanloc.xlsx', index_col=0)
channels1020 = np.array(data1020.index)
value1020 = np.array(data1020)

# 将电极通道名称和对应三维坐标位置存储为字典形式
list_dic = dict(zip(channels1020, value1020))
# print(list_dic)
# 封装为MNE的格式，参考原biosemi的存储格式
montage_1020 = mne.channels.make_dig_montage(ch_pos=list_dic,
                                             nasion=[5.27205792e-18, 8.60992398e-02, -4.01487349e-02],
                                             lpa=[-0.08609924, -0., -0.04014873],
                                             rpa=[0.08609924, 0., -0.04014873])

# 图示电极位置
# montage_1020.plot()
# plt.show()

# montage = mne.channels.read_custom_montage(montage_1020)
# print(montage)
info = mne.create_info(ch_names=montage_1020.ch_names, sfreq=160, ch_types='eeg')

# 画图
# fig, ax = plt.subplots(ncols=3, figsize=(4, 4), gridspec_kw=dict(top=0.9), sharex=True, sharey=True)
# fig = plt.figure()
# fig.patch.set_facecolor('blue')
# # fig.patch.set_alpha(0)
# ax1 = fig.add_axes([0.5,0.5,0.5,0.5])

for i in range(1):

    # plt.subplot(1, 1, i + 1)
    evoked = mne.EvokedArray(output64.detach().numpy()[i,:].reshape(64, 1), info)
    # print(eeg_re_prm[:, i].reshape(31, 1))
    # print()
    evoked.set_montage(montage_1020)
    im, cm = mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, show=False, cmap='viridis')

    plt.title("{}".format(i+1))
    # 添加所有子图的colorbar，
    # plt.colorbar(im)
    plt.show()
    # name = os.path.splitext(eeg_re_prm)[0]
    # plt.savefig("D:\pythonProject\svr/brainMap/{}".format(i+1))

    plt.close()