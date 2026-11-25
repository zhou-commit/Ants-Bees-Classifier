import torch.nn as nn
import torch
import torch.nn.functional as F

# 演示卷积操作的过程，本质上就是在stride为1的情况下，kernel在输入张量上滑动，每次滑动都进行卷积运算
input = torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]])

Kernel = torch.tensor([[1,2,1],  # 卷积核的定义
                      [0,1,0],
                      [2,1,0]])

input = torch.reshape(input, (1, 1, 5, 5))    #进行维度的扩充
Kernel = torch.reshape(Kernel, (1, 1, 3, 3))

output1 = F.conv2d(input, Kernel,stride=1) #conv2d的输入必须是4维张量
print(output1)
output2 = F.conv2d(input, Kernel,stride=2) #stride为2时则卷积核在输入张量上滑动的步长为2，得到2×2的输出张量
print(output2)
output3 = F.conv2d(input, Kernel,stride=1,padding=1) #stride为3时则卷积核在输入张量上滑动的步长为3，得到1×1的输出张量
print(output3)
