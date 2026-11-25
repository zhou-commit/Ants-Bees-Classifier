from gradio.components import label
from torch.utils.data import DataLoader
from data_read import MyDataset
import matplotlib.pyplot as plt
'''''数据加载的一种形式，就是把data_read的Mydataset对象拿过来用，但是你自己在这边要定义一个Mydataset'''
# 加载数据方式二： from data_read import train_dataset,然后使用dataloader加载
import cv2

'''举例，先加载数据集'''
train_dataset = MyDataset(r"origin_data/train", "ants_image", "ants_label")   #调用已存在包里面的类
train_loader = DataLoader(dataset=train_dataset, batch_size=3, drop_last=True,shuffle=True, num_workers=0)
# train_loader是一个Dataloader的一个实例对象，其产出的结果是调用getitem返回的[二元组]，即图像和标签

# val_dataset = MyDataset(r"origin_data/val", "ants_image",)
# val_loader = DataLoader(dataset=val_dataset, batch_size=3, shuffle=true, num_workers=0)
print(len(train_loader))      # len得到的是batch的个数，即批数

#图像显示
for imgs, labels in train_loader:
    print("一个batch的图像张量形状:", tuple(imgs.shape))   # (3, 3, H, W)
    print("一个batch的标签:", labels.tolist())

    img0 = imgs[0]                          # 取这批第1张，(C,H,W)，值域[0,1]
    img0 = img0.permute(1, 2, 0).numpy()    # CHW -> HWC，并转成 numpy
    plt.imshow(img0)                        # matplotlib 默认按 RGB 显示
    plt.title(f"label={labels[0].item()}")  # .item() 把单元素张量取成普通数字
    plt.axis('off')
    plt.show()
    break                                   # 调试阶段:测试完batch 就停，防止无限刷屏





