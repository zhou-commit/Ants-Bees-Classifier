import torch, torchvision

'''----加载方式1----'''
# model = torchvision.models.vgg16()
# # 本项目使用到的vgg模型只加载最后一层classifier的输出结果，所以只加载classifier的权重参数
# model.classifier[6] = torch.nn.Linear(4096, 2)
# model.load_state_dict(torch.load('vgg16_method1.pth',weights_only=True ))
# print(model)

'''----加载方式2----'''
# 将整个文件加入到白名单中，而不仅仅是加载权重参数,然后再重新加载模型
torch.serialization.add_safe_globals([torchvision.models.vgg.VGG])
model = torch.load('vgg16_method1.pth', weights_only=True)
print(model)                           # 输出的全部都是字典模式存储的参数
