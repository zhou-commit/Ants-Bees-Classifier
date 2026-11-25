import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_read import MyDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

# 1,先加载数据集
train_dataset = MyDataset(r"origin_data/train", "ants_image", "ants_label")
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=0)

# 2,定义模型(一个简单的卷积神经网络)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)

        # 最大池化层的定义，目的就是为了减少训练量从而加速训练，同时保留重要的特征信息
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, ceil_mode=True)

        '''定义全连接层，输入维度为6*113*113（通道数（C） × 特征图高（H） × 特征图宽（W）），卷积池化后的特征图
        展平维度，输出维度为512,但是因为pool层进行了stride=2的池化操作，根据计算结果pool输出的特征图尺寸为113*113'''
        self.fc1 = nn.Linear(in_features=6*113*113, out_features=512)
        self.relu = nn.ReLU()                  # 定义激活函数ReLU，目的是
        self.fc2 = nn.Linear(in_features=512, out_features=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) 
        # 进入全连接层之前需要展平，其中x.size(0)是batch_size，-1是神经网络将C*H*W自乘进行运
        # 其他写法：x = x.view(-1, 6*113*113) or x = torch.flatten(x, 1)//从第一维开始展平

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()

# 3,测试显示图像卷积效果
writer = SummaryWriter('logs/run2')

def show_feats(x):
    return make_grid(x[0].unsqueeze(1), nrow=6, normalize=True)
'''make_grid函数可以将多个图像拼接成一个图像,(Tensor张量，nrow每行图片数，normalize是否归一化到[0,1],为了防止因为
    截断而导致图片纯黑，所以进行最大最小值归一化处理),unsqueeze()将(C,H,W) -> (C,1,H,W)，就是让 make_grid 明白
    是 C 张单通道图，而不是 1 张 C 通道图'''

step = 0
for images,labels in train_loader:
    conv_output = model.conv1(images)
    pool_output = model.pool(conv_output)
    # print(len(output))            # 输出是32，表示32张图片，对任何张量，len(x) 返回的就是它第 0 个维度的大小
    flat = pool_output.view(pool_output.size(0), -1)
    liner_output = model.fc1(flat)
    relu_output = model.relu(liner_output)
    
    # print(images.shape)
    '''1.结果是[32, 3, 224, 224]，表示32张图片，每张图片经过卷积层后有3个特征图，每个特征图的大小是224x224
    记住PyTorch里图像数据的第 0 维永远是 batch（一批里有几张样本，记作 N）;后面才是通道C、高H、宽W
    2.模型里几乎任何一层，batch 维(第0维)都原样不动;卷积/池化只改通道、高、宽，
    3.通道由__getitem__ 里 .convert('RGB')决定，高宽由compose里的centercrop决定'''

    # conv_output = conv_output[0]                            # output[0]是一个三维张量，因为按索引去batch_size=0的下标，batch的维度就被去除了

    '''
    writer.add_image('原图：',make_grid(images, nrow=8, normalize=True) , step, dataformats='CHW')

    # 放弃reshape旧方法--conv_output = conv_output.reshape(6,1, 224, 224)      # 看第一张图的6个特征图，即卷积效果，可以看到最后一张图为黑色，是因为特征图的值几乎都小于零，所以被截断成0，显示纯黑
    writer.add_image('卷积后的特征图', show_feats(conv_output), step, dataformats='CHW')
    writer.add_image('池化后的特征图', show_feats(pool_output), step, dataformats='CHW')  # 注意：dataformats有几个字母就必须对应几维的张量
    writer.add_scalar('ReLU曲线', relu_output.mean(), step)
    step += 1

    # 注意：全连接输出是「特征向量」（一串数，没有图像意义，不该用 add_image 看）
writer.close()
'''

# 图像张量形状的变化
print(images.shape)
print(conv_output.shape)
print(pool_output.shape)
print(liner_output.shape)
print(relu_output.shape)
print('-----------------------------')

# 自定义一个手动调节的优化器，model.parameters代表模型的参数，这个一定要有
opt = torch.optim.SGD(model.parameters(), lr=0.01)  
loss = nn.CrossEntropyLoss()
p = model.fc2.bias                          # bias代表的是偏置值，即b
before = p.detach().clone()                 # 更新前的权重参数
print("更新前权重：",before)                 # 输出两个权重参数对应两个类别

for epoch in range(10):                     # 设置学习的轮次
    running_loss = 0.0
    for images,labels in train_loader:          # 每个batch32张图，实际上的完整批次就3次，因为drop_last=True,每次也只进行一轮的学习
        opt.zero_grad()                         # ①清空上一次的梯度，防止梯度叠加
        model_output = model(images)            # ②前向传播计算预测
        result_loss = loss(model_output, labels)
        # print(result_loss)
        '''输出的每一行是32个样本的loss值的均值，默认reduction='mean'，所以把这一批 32 张图各自的损失平均成 1 个标量再返回
        如果 reduction='sum'，则把这一批 32 张图各自的损失求和再返回，如果 reduction='none'，则分别返回 32 个样本的损失，
        CE衡量的是对模型正确的信任度，而不是真实标签概率与预测概率之间的差异'''

        result_loss.backward()                 # ③更新梯度，即dw，存进p.grad
        opt.step()                             # ④更新权重w，即p=p-lr*p.grad
        after = p.detach().clone()             # 更新后的权重参数
        running_loss += result_loss.item()
        print(f'Epoch {epoch+1}, Loss: {result_loss.item():.4f}')
        # 最终应该是可以观察到loss在不断减小的，无论grad取正值还是负值，△L=-lr*grad²，所以loss会不断减小

print('-----------------------------')
print("更新后权重：",after)
print("最终loss的加和：", running_loss)         # 其实就是最后一个epoch的三个batch的result_loss的和

git config --global --list