"""
MyDataset.py
数据加载模块,用来加载数据
"""

import torch
import torch.nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
"""
torch.utils.data提供了向网络中传递数据的标准pipeline，一般通过Dataset将各种类型（.txt .csv .png .pth）的数据处理成tensor，
用DataLoader把处理完的tensor传入网络进行训练，一般只需要自定义Dataset即可
"""

class MyDataset(Dataset):
    def __init__(self, file_path, data_type='train'):
        self.data_type = data_type
        # 传入CSV文件的地址并用pandas读取
        df = pd.read_csv(file_path)
        # 把性别转换成二值变量
        if self.data_type == 'train':  # 判断是训练数据还是测试数据
            df.loc[df['gender'] == 'Male', 'gender'] = 0
            df.loc[df['gender'] == 'Female', 'gender'] = 1
        # 把与处理完的data转换成numpy.array并传给self
        self.data = df.values.astype(float)

    # __getitem__ 是自定义Dataset时必须写的第一个函数，也是整个Dataset的核心，Dataloader会通过这个函数读入tensor
    def __getitem__(self, index):  # index表示读入第几个样本，必须写
        if self.data_type == 'train':
            DataX = self.data[index,:-1]  # 读取第index个样本的自变量
            DataX = torch.tensor(DataX, dtype=torch.float32)  # 转化为tensor
            DataY = self.data[index, -1]  # 读取第index个样本的因变量
            DataY = torch.tensor(DataY, dtype=torch.float32)  # 转化为tensor
            return DataX, DataY
        else:
            DataX = self.data[index, :]  # 读取第index个样本的自变量
            DataX = torch.tensor(DataX, dtype=torch.float32)  # 转化为tensor
            return DataX

    # __len__ 是自定义Dataset时必须写的第二个函数，用来表示index最大值，也就是样本量
    def __len__(self):
        return int(self.data.shape[0])


if __name__ == '__main__':
    # 用Dataset读取训练数据
    dataset = MyDataset(r'..\data\gender_classification_train.csv')
    # 把Dataset装入DataLoader，每次读取512个样本，打乱样本顺序
    DL = DataLoader(dataset, batch_size=512, shuffle=True)
    # DataLoader 是一个python迭代器，一般用for循环访问，也可以用以下方法单次访问
    data, target = next(iter(DL))
    print(data.shape, target.shape)