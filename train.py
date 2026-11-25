import torch,os,random
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset,ConcatDataset
from data_read import MyDataset
'''完整的模型训练套路'''

'''1.设计随机种子'''
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False       # 模型的卷积层预先优化关闭
    torch.backends.cudnn.deterministic = True    # 确定为默认卷积算法

set_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''2.定义构造数据集的方法'''
val_transform = transforms.Compose([             # 这个是在数据集进行验证时需要进行的额外的数据增强
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_transform = transforms.Compose([           # 训练数据集需要进行的数据转换
    transforms.RandomResizedCrop(224),           # 随机裁剪，用于数据增强
    transforms.RandomHorizontalFlip(),           # 随机水平翻转，用于数据增强
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# 构建train数据集
ants_dataset = MyDataset(root_dir='./origin_data/train',img_dir='ants_image',label_dir='ants_label',transform=train_transform)
bees_dataset = MyDataset(root_dir='./origin_data/train',img_dir='bees_image',label_dir='bees_label',transform=train_transform)
train_dataset = ConcatDataset([ants_dataset,bees_dataset]) 
# 注意：因为合并的数据集数据较少，如果batch_size取32可能单批次取到的都是ant或bee，而subset无法进行均衡抽样，导致冒烟测试训练效果不好

# 构建val数据集
val_dataset = ImageFolder(root='./origin_data/val',transform=val_transform)
# print(val_dataset.classes, val_dataset.class_to_idx)  # 打印验证集的类别和类别索引

'''3.训练数据集'''
def train_model(model,dataloader,criterion,optimizer,val_loader=None,num_epoch=20):
    # 创建一个checkpoint来保存模型的目录
    ckpt_dir = './checkpoint'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    best_acc = 0                       # ckpt里面会有每轮训练的acc以及最佳acc

    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0             # 单批样本的损失
        correct = 0                    # 预测正确的样本数量
        total = 0                      # 参与统计的样本总数

        # 使用tqdm来显示训练进度条
        with tqdm(dataloader,unit='batch') as tepoch:     # 从dataloader中取出相应数据到tepoch
            tepoch.set_description(f'Epoch {epoch+1}/{num_epoch}')
            for i,(inputs,labels) in enumerate(tepoch):
                inputs,labels = inputs.to(device),labels.to(device)

                # 梯度清零-前向传播-计算损失-反向传播更新权重-更新模型的参数
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()                   # 全部损失的加和
                _,predicted = outputs.max(1)                  # 获取预测结果,predicted是预测的类别
                total += labels.size(0)                       # 获取参与统计的样本总数
                correct += predicted.eq(labels).sum().item()  
                '''累加所有预测正确的概率，output是模型的输出（A，B）分别是蚂蚁和蜜蜂的分数，还没概率化，
                而.max是取最大值和下标，用_,predicted只取下标得到的是预测类别[0.1.0.1...],然后与labels进行比较，
                相同的就是预测正确的，sum().item()是将这些预测对的数量加和，correct/total自然得到acc'''

                accuracy = 100. * correct/total                            # 将预测率转化成百分数
                tepoch.set_postfix(loss=running_loss/(i+1),acc=accuracy)   # 更新进度条显示,i是下标，running_loss是累加的损失，故loss可以表示训练进度
                tepoch.update()

        # 每个epoch模型结束后需要再验证集上进行评估
        if val_loader is not None:
            val_acc, val_loss = evaluate(model,val_loader)                 # 采用元组解包的方式获取验证集的acc和loss
            # 评估验证集与训练集的差异
            print(f"Epoch{epoch+1}/{num_epoch} train_loss={running_loss/(i+1):.4f} " 
                f"train_acc={accuracy:.2f} val_acc={val_acc:.2f} val_loss={val_loss:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model,os.path.join(ckpt_dir,'best.pt'))         # 只保留验证集上最优的模型
        else :
            print(f"Epoch{epoch + 1}/{num_epoch} train_loss={running_loss / (i + 1):.4f} " f"train_acc={accuracy:.2f}")

        torch.save(model,os.path.join(ckpt_dir,'cp{}.pt'.format(epoch)))   # 保存每一个epoch模型以及其训练的最优acc

def evaluate(model,val_loader):
    model.eval()                    # 设置模型为评估模式
    correct = 0
    total = 0
    val_loss = 0.0                   # 单批验证集样本的损失

    with torch.no_grad():            # 不做反向传播、不记录梯度，因为验证数据集就是用的新的数据集对模型进行评估，已经无需学习了，只需要验证准确率怎么样就行
        for inputs,labels in val_loader:
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs,labels).item() * labels.size(0)   # 累加验证集的批损失
            _,predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100*correct/total if total else 0,val_loss/total

'''4,构建模型'''

# 加载resnet18模型进行预训练，提取模型的通用特征值来跑冒烟测试,models.ResNet18_Weights代表使用resnet的自带的参数
resnet18= models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

num_features = resnet18.fc.in_features                # 获取resnet18模型的最后一次输出
resnet18.fc = nn.Linear(num_features,2)               # 替换最后一层的全连接层，将输出的类别数设置为2
FREEZE_BACKBONE = True                                # 冻结开关的开启  

# 冻结卷积骨干，固定这些参数不参与训练，直接复用它们提取的通用特征
if FREEZE_BACKBONE:
    for param in resnet18.parameters():               # 冻结所有参数
        param.requires_grad = False
    for param in resnet18.fc.parameters():            # 单独解冻fc层的参数
        param.requires_grad = True

'''5.定义数据集的加载方式'''
# 进行冒烟测试，取少量样本和小epoch，看看模型能否跑通
SMOKE_TEST = False
SMOKE_N = 32

if SMOKE_TEST:
    # 在冒烟时进行抽样准备
    n_sum = len(ants_dataset)+len(bees_dataset)
    n_val = len(val_dataset)
    n_idx1,n_idx2 = list(range(n_sum)),list(range(n_val))
    random.shuffle(n_idx1)                                       # 随机打乱索引
    random.shuffle(n_idx2)

    idx1 = n_idx1[:SMOKE_N]                                      # 让子集包含两个类别，随机取SMOKE_N个样本
    train_data = Subset(train_dataset,idx1)                      # Subset（原始数据，索引列表）
    idx2 = n_idx2[:SMOKE_N//2]
    val_data = Subset(val_dataset,idx2)                          # 验证集的冒烟测试应该采用更小的索引范围
else:
    train_data = train_dataset
    val_data = val_dataset

# 创建训练集和验证集的数据加载
train_loader = DataLoader(train_data,batch_size=32,shuffle=True,num_workers=0,drop_last=False)
val_loader = DataLoader(val_data,batch_size=32,shuffle=False,num_workers=0,drop_last=False)

'''6.定义损失函数和优化器'''
criterion = nn.CrossEntropyLoss()
if FREEZE_BACKBONE:
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, resnet18.parameters()),lr=1e-3,weight_decay=1e-4)
        # filter(...)过滤出requires_grad为True的参数,lambda表示自定义函数p->p.requires_grad
        # 无weight权重会越来越大，模型越来越自信，有weight_decay可以防止模型过拟合
else:
    optimizer = torch.optim.Adam(resnet18.parameters(), lr=1e-4)

if __name__ == '__main__':
    # 启动训练，先smoke->train
    resnet18.to(device=device)
    epoch = 2 if SMOKE_TEST else 20           # 如果是冒烟测试则跑两轮，否则完整测试跑10轮
    train_model(resnet18,train_loader,criterion,optimizer,val_loader=val_loader,num_epoch=epoch)










