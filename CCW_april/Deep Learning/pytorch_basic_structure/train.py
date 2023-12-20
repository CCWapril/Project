"""
train.py
训练网络的主循环脚本
"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.MyDataset import MyDataset
from network.MyNetwork import MyNetwork
from network.MyLoss import MyLoss


def train_epoch(epoch, model, criterion, optim, train_dataloader):
    model.train()
    # 在1个epoch中，数据被分为多个batch，而不是一次全部使用，因为少量多次的算loss（mean cost）可以让梯度更具多样性，
    # 这种多样性往往能帮助优化器跳出局部最优，前面定义的 batch size 就是这里一次读入的数据量
    for batch_idx, (inputs, outputs) in enumerate(train_dataloader):
        # 优化器 清零
        optim.zero_grad()
        # 神经网络正向传播
        pred = model.forward(inputs)
        # 计算loss并保存
        loss = criterion(torch.flatten(pred), outputs)
        # 计算反向传播的梯度
        loss.backward()
        # 优化器更新网络参数
        optim.step()
        # print训练结果
        if batch_idx % 100 == 0:
            dataset_size = len(train_dataloader)
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, dataset_size,
                100. * batch_idx / dataset_size, loss))


def eval_epoch(model, data_loader, criterion):
    model.eval()
    pred_list = []
    true_list = []
    for batch_idx, (inputs, outputs) in enumerate(data_loader):  # 当eval set的数据量较大时，可以使用mini batch测试结果再通过cat保存
        # 用正向传播进行预测
        pred = model.forward(inputs)
        pred = torch.flatten(pred)
        pred_list.append(pred)
        true_list.append(outputs)

    pred = torch.cat(pred_list)
    true = torch.cat(true_list)
    # 计算 eval set 的loss
    loss = criterion(pred, true)
    # 将数据类型转化成numpy，计算acc
    pred = pred.detach().numpy()
    pred = np.where(pred>0.5,1,0)
    true = true.detach().numpy()
    # 保存eval结果
    Acc = np.mean(pred == true)
    print('Test set: Average loss: {:.4f}, Test Accuracy: {:.2f}%,'.format(loss, 100.*Acc))


def predition(model,test_dataset):
    model.eval()
    pred_list = []
    for batch_idx, inputs in enumerate(test_dataset):  # 当eval set的数据量较大时，可以使用mini batch测试结果再通过cat保存
        # 用正向传播进行预测
        pred = model.forward(inputs)
        pred = torch.flatten(pred)
        pred_list.append(pred)
    pred = torch.cat(pred_list)
    pred = pred.detach().numpy()
    pred = np.where(pred > 0.5, 'Female', 'Male')
    pred_df = pd.DataFrame({'gender':pred})
    pred_df.to_csv('result.csv',index=False)


if __name__=="__main__":
    # 读取数据
    train_dataset = MyDataset(r'data\gender_classification_training_set.csv')
    eval_dataset = MyDataset(r'data\gender_classification_eval_set.csv')
    test_dataset = MyDataset(r'data\gender_classification_test.csv', data_type='test')
    # dataset读取数据后装进dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=512, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # 初始化网络
    model = MyNetwork(7,[8,8,8],1)

    # 初始化优化器，学习率为10e-5，常用的两种优化器为 Adam和SGD, 大多数情况下两种都能使用，一般Adam收敛速度较快
    optim = torch.optim.Adam(model.parameters(), lr=10e-5)
    # optim = torch.optim.SGD(model.parameters(), lr=10e-5)

    # 定义损失函数，可以自己定义也可以选择 torch.nn 中的预设, 数据、网络结果、损失函数三者必须匹配
    # criterion = MyLoss()
    # criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    # 开始训练
    for epoch in range(100):
        train_epoch(epoch, model, criterion, optim, train_dataloader)
        eval_epoch(model, eval_dataloader, criterion)
    # 保存预测结果
    predition(model, test_dataloader)
    # 保存结果
    torch.save(model.state_dict(), "./checkpoints/TrainedModel.ckpt")
    print('model saved')

    # 测试