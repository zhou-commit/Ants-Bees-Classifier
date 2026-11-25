import torch
import torch.nn as nn

class Seq_use(nn.Module):
    def __init__(self):
        super(Seq_use, self).__init__()

        self.models = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,padding=2),
            nn.MaxPool2d(kernel_size=2,stride=1),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,padding=2),
            nn.MaxPool2d(kernel_size=2,stride=1),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding=2),
            nn.MaxPool2d(kernel_size=2,stride=1),
            nn.Flatten(),
            nn.Linear(in_features=64*29*29,out_features=64),
            nn.Linear(in_features=64,out_features=10),
        )
        '''要代入H(out)的计算公式去进行计算，并且注意通道数的更新
        卷积与池化都是默认padding=0，dilation=1，但是卷积默认stride=1，池化默认stride=kernel_size，ceil_mode=False(若为true则向上取整)
        linear的输入必须是2维的，因此需要将多维的输入张量展，且要进行维度匹配'''

    def forward(self, x):
        return self.models(x)

model = Seq_use()
print(model)
input = torch.ones((64,3,32,32))  
# 创建全1张量（batch，C，H，W）->创建64张32×32的RGB彩色全1图片张量,用来测试模型的前向传播或者进行占位初始化
output = model(input)
print(output.shape)
# 计算出来的结果是[64,10]，即64张图片的10个分类结果
