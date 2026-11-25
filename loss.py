import torch
import torch.nn as nn

class loss_test(nn.Module):
    def __init__(self):
        super(loss_test, self).__init__()

    input = torch.tensor([1,2,3], dtype=torch.float32)
    target = torch.tensor([1,2,5], dtype=torch.float32) # dtype的目的是将张量转化为float32的数据类型

    loss = nn.L1Loss()(input, target)                   # 计算输入和目标之间的L1损失，即绝对值差的平均值
    MSE = nn.MSELoss()(input, target)

    x = torch.tensor([0.1,0.2,0.3])
    y = torch.tensor([1])                               # 意思是y是一维分数，背后softmax会将logits->概率分数，并且三个分类概率相加为1
    x = torch.reshape(x, (1, 3))                        # 其实就是将x的维度从1->2，然后从一行三个数变成一行三列三个数
    CE = nn.CrossEntropyLoss()(x, y)                    # 不能将回归向量塞给分类损失函数计算loss，要不然就会先进行softmax归一化换算成概率再进行计算
    print(CE)
