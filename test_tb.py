'''TensorBoard的使用'''
from torch.utils.tensorboard import SummaryWriter
import torch
import math
import numpy as np                          # numpy用于数组计算，该处是为了转化为numpy数组
from PIL import Image
from torchvision import transforms          # torchvision.transforms用于图像的转换,转化为张量

writer = SummaryWriter("logs")              #如果想将不同的数据写入不同的logs文件夹中，可以创建不同的Writer对象，如w1, w2, w3...
# writer.add_image("image", img)            # add_image()函数用于查看单张图片，第一个参数是标签，第二个参数是图片

# 设置一个sigmoid函数
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

for i in range(200):
    # writer.add_scalar("y=2x", 2*i, i)
    '''writer.add_scalar(tag, scalar_value, global_step)，其中tag: 标签，曲线名称
    scalar_value: 标量值，y轴要记录的值；global_step: 步数，x轴位置，通常是训练的轮数'''

    # writer.add_scalars("y=3x and y=4x", {"y=4x": 4*i, "y=3x": 3*i}, i)     
    '''1. 如果想要将多个曲线放在一个图像里面，可以采用字典dict{}的方式,dict{key:value}，key是曲线的名称，value是曲线的值'''
    
    x = i/10 - 10                                                   #注意sigmoid函数的输入的表示范围
    y = 1/(1 + math.exp(-x))
    writer.add_scalar("sigmoid", y+0.05*np.random.randn(), i)       #sigmoid函数的表示，0.05*np.random.randn()模拟随机噪声分布

'''2.PIL→numpy'''
image_path = r"origin_data\train\ants_image\0013035.jpg"  # 替换为图片的路径 
img = Image.open(image_path)
print(type(img))  # 打印图片的类型
img_array = np.array(img)
print(type(img_array))
writer.add_image("image", img_array, 1, dataformats='HWC')
# 可以改变step的值，查看不同阶段的图片，都会在tensorboard里面显示

'''3.PIL→tensor'''
img_tensor = transforms.ToTensor()(img)
writer.add_image("image", img_tensor, 1, dataformats='CHW')

writer.close()