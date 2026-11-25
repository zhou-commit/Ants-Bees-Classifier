from torchvision import transforms
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# 1.第一种使用方法，totensor()函数的使用，将PIL转换为Tensor
img_path = r"origin_data\train\ants_image\0013035.jpg"
img = Image.open(img_path).convert('RGB')
img_tensor = transforms.ToTensor()(img)                          #img_tensor[通道][行][列]就是简写为CHW
print(img_tensor)

writer = SummaryWriter('logs/run1')
# 将PIL转换为Tensor或者 array
# img_array = np.array(img)
# writer.add_image('image', img_array,2,dataformats='HWC')
writer.add_image('toTensor图像显示为：', img_tensor,1,dataformats='CHW')        #采用Tensor直接来进行转换

# 2.第二种使用方法,normalize的使用，属于数据预处理的一步
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.4,0.3,0.2])
'''采用标准化对图像进行特征缩放，公式为：(x - mean) / std，normalize（mean, std）中的mean和std分别表示
图像的均值和标准差，如果计算出来标准化的张量为负数，则不显示，标准化的可显示结果一定落在[0,1]之间'''

img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image('normalize图像显示为：', img_norm,1,dataformats='CHW')

# 3.第三种使用方法，resize的使用，属于数据预处理的一步
print(img.size)
trans_resize = transforms.Resize((512,512))                    #将数据缩放到指定尺寸
img_resize = trans_resize(img_tensor)
writer.add_image('resize图像显示为：', img_resize,1,dataformats='CHW')

# 4.第四种使用方法，compose的使用，compose([])是列表,属于数据预处理的一步
trans_compose = transforms.Compose([
    transforms.Resize(512),                         #将数据缩放到指定尺寸，如果resize的参数只有一个，那么就是将短边缩小到该参，长边按照短边缩放比例缩小
    transforms.CenterCrop(512),                     #将数据进行中心裁剪，
    transforms.ToTensor(),
    # transforms.Normalize([0.5,0.5,0.5],[0.4,0.3,0.2])  一般的compose流程会加上张量转换和归一化处理
    ])
img_compose1 = trans_compose(img)
writer.add_image('compose图像显示为：', img_compose1,1,dataformats='CHW')

# 5.第五种使用方法，randomcrop的使用，属于数据预处理的一步,一般用于训练集的数据处理
trans_randomcrop = transforms.RandomCrop(256)
for i in range(5):                                 #利用for循环去实现随机裁剪的显示，从step1-step5
    img_compose2 = transforms.Compose([trans_randomcrop, transforms.ToTensor()])(img)
    writer.add_image('randomcrop图像显示为：', img_compose2,i,dataformats='CHW')

writer.close()




