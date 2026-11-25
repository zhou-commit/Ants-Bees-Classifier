from torch.utils.data import Dataset
import cv2
from PIL import Image                       #如果不会使用cv2包就用这个
import os
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt             #用matplotlib显示图片
from torchvision import transforms

class MyDataset(Dataset):    #定义一个数据类，用来进行数据的读取
    def __init__(self,root_dir,img_dir,label_dir,transform=None):    #在初始化函数中进行量的设置,self代表实例(dataset)自身，用来给对象绑定属性
        self.root_dir = root_dir
        ''' self.root_dir = root_dir 的意思是：把后面传入的 "data/train" 这个值，挂到 dataset 这个对象身上，
        并给它起个名字叫 root_dir，这样就是为了区分于内外部的dataset，以后在别的地方调用self.root_dir可以直取 '''
        self.label_dir = label_dir
        self.img_dir = img_dir                         #一个为了读图片一个为了读标签
        self.path = os.path.join(self.root_dir,self.img_dir)

        '''图片过滤：只保留那些“同名 .txt 标签文件存在”的图片文件'''
        self.img_path = [f for f in os.listdir(self.path)     
            if os.path.exists(os.path.join(self.root_dir, self.label_dir,os.path.splitext(f)[0] + ".txt"))] 
             
        self.class_to_idx = {"ants": 0, "bees": 1}   #文字→数字表
        '''这个的意义就是如果外部传入了transform就用外部传入的transform，如果没有就用这个默认的transform'''
        self.data_transform = transform if transform is not None else transforms.Compose([
            transforms.Resize(256),        # 短边缩到256（保比例，此时每张长边仍不等）
            transforms.CenterCrop(224),    # 中心裁 224x224 → 尺寸全相等
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def get_img_path(self,index):                    #将拼接形成图片完整路径做成一个单独的方法
        img_name = self.img_path[index]              #获取图片名称
        return os.path.join(self.path,img_name)      #拼接形成完整的图片的路径，精确到某个图片的具体位置

    def __getitem__(self, idx):                      #创建一个图片地址的列表进行图片索引
        img_item_path = self.get_img_path(idx)       #获取图片的路径,直接调用git_img_path方法
        self.img_item_path = img_item_path
        img = Image.open(img_item_path).convert('RGB')
        img_tensor = self.data_transform(img)
        
        # 读标签文字，并且使用class_to_idx将标签转化为整形数字
        label_name = os.path.splitext(self.img_path[idx])[0] + ".txt"
        label_path = os.path.join(self.root_dir, self.label_dir, label_name)
        with open(label_path, "r", encoding="utf-8") as f:
            cls = f.read().strip()                  # strip() 去掉换行/空格，得到干净的 "ants"
        label = self.class_to_idx[cls]

        '''如果有imagefloder，train_dataset = ImageFolder(root=train_path, transform=data_transform)
        ImageFolder自动遍历子文件夹：标签由文件夹名决定（按字母序 cat=0, dog=1）,那么就可以直接使用class_to_idx
        label = self.class_to_idx[cls]可以直接将标签表示出来'''
        return img_tensor,label

    def __len__(self):
        return len(self.img_path)

    def preview(self,idx):                           #用pil预览图片用来检测路径是否正确
        img_path = self.get_img_path(idx)
        print("图片的路径是：",img_path,"图片的长度是：",len(self))
        img = Image.open(img_path).convert('RGB')
        img.show()
        return img

    def label_show(self,idx):
        label_name = os.path.splitext(self.img_path[idx])[0] + ".txt"   # 把图片文件名的后缀换成 .txt
        label_path = os.path.join(self.root_dir,self.label_dir,label_name)
        with open(label_path, "r", encoding="utf-8") as f:
            content = f.read()
        print("该图片是一个：",content)
        return content

if __name__ == '__main__':    
    #__name__是Python中一个特殊的变量，它代表了当前模块的名字,如果从别的模块导入则视作模块名，就能做到「自己跑才执行、被导入时不执行」
    ants_dataset = MyDataset(r"origin_data/train", "ants_image", "ants_label")
    bees_dataset = MyDataset(r"origin_data/train", "bees_image", "bees_label")

    '''1.使用PIL预览图片'''
    # ants_dataset.preview(0)
    # ants_dataset.label_show(0)

    train_dataset = ConcatDataset([ants_dataset, bees_dataset])               #将两个数据集合并成一个数据集
    # print(len(train_dataset))


    '''2.使用cv2预览图片'''
    # img = cv2.imread(ants_dataset.get_img_path(0))
    # cv2.imshow('image', img)                #OpenCV 的 imshow 必须配合 waitKey 才能刷新图像界面，destroyAllWindows用于关闭所有窗口
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print("灰度图像形状:", image.shape)
    # print("灰度图像像素值:")
    # print(image)

    '''3.使用matplotlib预览图片'''
    img = cv2.imread(ants_dataset.get_img_path(0))
    plt.imshow(img)                         #imshow()函数用于显示图像，plt.axis('off')用于关闭坐标轴
    plt.axis('off')
    plt.show()
