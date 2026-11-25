import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import models
from data_read import MyDataset
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)

# ---------- 1) 数据：复用你的 MyDataset，ants + bees 拼成训练集 ----------
# 提醒：VGG 用 ImageNet 归一化，建议把 MyDataset.data_transform 换成下面这样
# （若不改，也能跑，只是精度略差；改法见第四节）
ants  = MyDataset(r"origin_data/train", "ants_image", "ants_label")
bees  = MyDataset(r"origin_data/train", "bees_image", "bees_label")
train_dataset = ConcatDataset([ants, bees])     # 标签：ants=0, bees=1
train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=0)

# ---------- 2) 模型：预训练 VGG16 + 冻结卷积 + 换 2 类分类头 ----------
weights = models.VGG16_Weights.IMAGENET1K_V1    # 官方 ImageNet 预训练权重（新写法，替代 pretrained=True）
model = models.vgg16(weights=weights)

# 冻结 features（卷积主干）：固定这些参数不参与训练，直接复用它们提取的通用特征
for param in model.features.parameters():
    param.requires_grad = False

# 只把分类头最后一层 4096->1000 换成 4096->2（你的两类）
model.classifier[6] = nn.Linear(4096, 2)        
# classifier 共 0~6 层，其中13层卷积+3层全连接，第6层是原 1000 类输出，我只需要第6层然后将它换为 2 类

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ---------- 3) 损失 / 优化器：只优化「没被冻结」的参数 ----------
loss_fn = nn.CrossEntropyLoss()
trainable = [p for p in model.parameters() if p.requires_grad]   # 只有 classifier 会更新
opt = torch.optim.SGD(trainable, lr=0.001, momentum=0.9)         # 能更新的参数进行优化，微调用小 lr + 动量

writer = SummaryWriter("logs/vgg16")
# ---------- 4) 训练循环（和你 SimpleCNN 的四步一模一样） ----------
step = 0
for epoch in range(5):
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        opt.zero_grad()                       # ① 清梯度
        out  = model(images)                  # ② 前向：(B,2)
        loss = loss_fn(out, labels)           # 计算损失
        loss.backward()                       # ③ 反向：只算 classifier 的梯度
        opt.step()                            # ④ 更新权重
        writer.add_scalar("train/loss", loss.item(), step)
        # print(f"epoch{epoch+1} step{step} loss={loss.item():.4f}")
        step += 1
writer.close()

torch.save(model.state_dict(), "vgg16_method1.pth")      # 保存整个模型,只保存权重参数不保存整个模型代码，可以直接调用加载整个模型

# 总结：VGG 的角色 = 预训练特征提取器，借用训练结果然后微调；数据小→冻结只训头,数据大→小 lr 微调更多层。

