"""
介绍torch.tensor的使用
torch.tensor 的定义和 numpy.array 一样，但tensor内置了神经网络的函数和方法，使用tensor搭建的网络可以自动完成梯度求导和反向传播
"""

import torch

"""
https://pytorch.org/docs/stable/tensors.html
"""

# 1 tensor 定义: 基本和numpy.array一模一样
tensor1 = torch.tensor([[1,2,3],[6,7,8]], dtype=torch.float)  # tensor 的定义的方法和 numpy.array一样， 最好在定义时统一dtype，保证程序整体稳定
print('tensor1: \n', tensor1)

tensor2 = torch.rand([2,3], dtype=torch.float)  # 随机生成
print('tensor2: \n',tensor2)

tensor3 = torch.zeros([2,3], dtype=torch.float)  # 0生成
print('tensor3: \n',tensor3)

tensor4 = torch.ones([2,3], dtype=torch.float)  # 1生成
print('tensor4: \n',tensor4)

print('tensor.shape: \n', tensor1.shape)

# 2 tensor 计算： 公式太多了，能想到的计算都有，具体见文档
print('tensor加减法: \n',tensor1+tensor2)

print('tensor乘除法: \n',torch.mm(tensor1,tensor2.T))

# 3 tensor 转换： 可以和 numpy 非常方便的转换

print('tensor转回numpy: \n',tensor1.numpy())



