"""
MyLoss.py
自定义损失函数的脚本
"""


import torch
import torch.nn as nn

"""
https://blog.csdn.net/weixin_35757704/article/details/122865272
"""

class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):  # 必须定义一个以forward命名的函数，用来计算loss，整个loss必须全是tensor运算，才能自定计算梯度
        return -torch.mean(y*torch.log(x)+(1-y)*torch.log(1-x))


if __name__ == '__main__':
    # 检查损失函数是否可以正常运算
    criterion = MyLoss()

    pred = torch.rand((400,1),dtype=torch.float)
    true = torch.rand((400,1), dtype=torch.float)

    loss = criterion(pred,true)
    print(loss)
