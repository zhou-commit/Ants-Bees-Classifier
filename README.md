# Learn_torch —— 蚂蚁 vs 蜜蜂图像分类学习项目

一个围绕 **ants vs bees（蚂蚁 / 蜜蜂）二分类** 的 PyTorch 学习仓库。从最基础的卷积、张量变换起步，逐步搭到一条完整的迁移学习训练流水线（`train.py`）。仓库里的脚本大多是**独立的小演示**，每个聚焦一个 API 或概念，源码内注释已写得较细，本 README 只做导览。

## 环境依赖

- Python 3.12.14 + PyTorch + torchvision
- tqdm、matplotlib、opencv-python、Pillow、tensorboard

---

## 文件导览

### 核心流水线

- **`data_read.py`** —— 自定义 `MyDataset`。把"图像文件夹 + 同名 txt 标签"封装成标准 `Dataset`，`__getitem__` 返回 `(图像张量, 整数标签)`；标签用 `class_to_idx` 把 txt 里的 "ants/bees" 映射成 0/1。细节：只保留"同名 .txt 存在"的图片，避免读到孤立文件。
- **`dataloader.py`** —— 用 `DataLoader` 批量加载 `MyDataset`，并取一个 batch 用 matplotlib 显示，验证加载是否正确。
- **`train.py`** —— 本项目主程序，完整训练套路：随机种子 → 构造数据 → 建模 → 训练 → 验证 → 存档，并带冒烟测试开关。几个要点：训练/验证用两套 transform（增强只上训练集）；验证集用 `ImageFolder`，标签来自子文件夹名；ResNet18 迁移学习，可选冻结卷积骨干（先全冻再解冻 fc）；`evaluate` 返回 `(val_acc, val_loss)`。

### 模型实现

- **`simple_cnn.py`** —— 手写一个简单 CNN，逐层打印张量形状变化，并把卷积/池化特征图拼成网格写入 TensorBoard；末尾用更新前后权重对比证明参数确实被优化。
- **`Seq_use.py`** —— 用 `nn.Sequential` 堆卷积网络，重点演示全连接 `in_features` 如何靠输出尺寸公式逐级推算。
- **`vgg16_use.py`** —— 预训练 VGG16 迁移学习：冻结卷积主干、只把分类头最后一层换成 2 类并微调。经验：数据小→冻结只训头，数据大→小 lr 微调更多层。
- **`model_load.py`** —— 加载已保存权重的两种方式（`load_state_dict` 填参数 vs 加白名单后 `torch.load` 整个对象）。

### 概念演示

- **`nn_conv.py`** —— 用 `F.conv2d` 手动演示 `stride` / `padding` 如何改变输出尺寸。
- **`transforms_use.py`** —— `ToTensor / Normalize / Resize / Compose / RandomCrop` 逐个变换的可视化演示。
- **`loss.py`** —— 对比 `L1Loss / MSELoss`（回归）与 `CrossEntropyLoss`（分类）的输入要求差异。
- **`test_tb.py`** —— TensorBoard 入门：`add_scalar / add_scalars / add_image` 的基本用法。

## 数据目录结构

```
origin_data/
├── train/
│   ├── ants_image/  ants_label/   # label 为同名 .txt，内容 "ants"
│   └── bees_image/  bees_label/   # 内容 "bees"
└── val/
    ├── ants_image/  bees_image/   # 验证集用 ImageFolder，标签由文件夹名决定
```

## 运行

```bash
python train.py                 # 完整训练流水线
tensorboard --logdir logs       # 查看训练日志
```

本项目仅用于学习用途。这个项目将数据预处理和训练的过程拆分成小块进行分析试验，仅供初学者调试学习。 新人小白第一次在GitHub上进行项目开源部署，如果对你有帮助的话还请多多支持，点一个免费的star。