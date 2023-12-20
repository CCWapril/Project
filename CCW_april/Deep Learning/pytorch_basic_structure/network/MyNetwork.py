"""
MyNetwork.py
构建神经网络结构的脚本
"""

import torch
import torch.nn as nn  # nn: Neural Network https://pytorch.org/docs/stable/nn.html


class MyNetwork(nn.Module):
    def __init__(self, input_dim: int, hiden_dim: list, output_dim: int):  # 定义一个全连接神经网络，传入输入层、中间层、输出层的维度

        # 这几行其实没用到，但写出来是个好习惯
        super().__init__()  # 可以不用管super是什么意思，大多数情况下直接加上这行就行
        self.input_dim = input_dim
        self.hiden_dim = hiden_dim
        self.output_dim = output_dim

        # torch.nn 内置了丰富的单层网络结构，像搭积木一样一行一行堆起来就行，见文档 https://pytorch.org/docs/stable/nn.html
        # nn.Sequential 可以将多层网络组合在一起，方便管理以及forward
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hiden_dim[0]),  # 线性层
            nn.ReLU()  # 激活层
        )

        n = len(hiden_dim)
        self.hiden_layer = nn.Sequential()
        for i in range(n-1):
            # add_module可以向已经定义的sequential添加新层，和list的append一样
            self.hiden_layer.add_module('Linear'+str(i), nn.Linear(hiden_dim[i],hiden_dim[i+1]))
            self.hiden_layer.add_module('ReLU'+str(i), nn.ReLU())

        self.output_layer = nn.Sequential(
            nn.Linear(hiden_dim[-1], output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):  # 网络必须定义一个forward函数，表示传入的数据x如何在网络中正向传播，通常把__init__里定义的layer连在一起就行
        x = self.input_layer(x)
        x = self.hiden_layer(x)
        y = self.output_layer(x)
        return y


if __name__ == '__main__':
    # 初始化网络
    model = MyNetwork(10, [5,6,7,8,9], 1)
    # 查看网络结构
    print(model)
    # 定义一个tensor 用以检查网络是否可以正常的正向传播
    data = torch.rand(400,10,dtype=torch.float)
    # 正向传播
    out = model(data)
    print(out.shape)
