# 1. 导入库
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models,transforms

# 3. 读取并预处理图片
def get_image_path():
    image_dir=Path(__file__).parent/"images"                ##使用Path类读取当前py文件坐在文件夹下的images文件夹
    png_path=image_dir/"test.png"                           ##拼接出png文件路径
    jpg_path=image_dir/"test.jpg"                           ##拼接出jpg文件路径
    
    if png_path.exists():                                   ##使用Path类实例对象的exists方法分别检查是否存在
        return png_path
    if jpg_path.exists():
        return jpg_path
    
def load_image(image_path,transform):
    ##加载图片
    image=Image.open(image_path).convert("RGB")             ##使用open方法按image_path打开图片并将其转化为RGB三通道
    image_tensor=transform(image)                           ##使用transform转换为张量
    image_tensor=image_tensor.unsqueeze(0)                  ##增加一个维度batch_size
    return image_tensor

def build_transform():
    ##图片处理：先按比例将短边缩到256--->从中心裁一个224×224--->转换成张量--->正态化
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485,0.456,0.406),
            (0.229,0.224,0.225)
            )
        ]
    )


# 4. 加载 ResNet18 并去掉分类
def build_feature_extractor(device):
    weights=models.ResNet18_Weights.DEFAULT                 ##定义模型权重采用已经训练好resnet模型
    model=models.resnet18(weights=weights)                   ##调用models中resnet
    model.fc=nn.Identity()                                  ##取消最后的依据特征输出类别分数的分类层
    model=model.to(device)                                  ##送至device运行
    model.eval()                                            ##模型为评估模式
    return model

# 5. 前向传播并打印特征 shape
def main():
    device="cuda" if torch.cuda.is_available() else "cpu"   ##参考代码中定义了一个函数实现，直接定义设备即可
    image_path=get_image_path()                             ##读取图片路径
    transform = build_transform()
    image_tensor = load_image(image_path, transform).to(device)
    model = build_feature_extractor(device)

    with torch.no_grad():
        features=model(image_tensor)
    
    print(f"使用的设备：{device}")
    print(f"图片张量形状：{image_tensor.shape}")
    print(f"特征形状：{features.shape}")
    print(f"前10个特征值:{features[0,:10].cpu()}")

if __name__=="__main__":
    main()


