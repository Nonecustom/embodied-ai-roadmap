##学习trainreference后手敲train
# 1. 导入库
import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import sys
import io
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 2. 定义 CNN 模型
#创建SimpleCNN类继承父类nn.Module
class SimpleCNN(nn.Module):                         
    def __init__(self):                            ##定义初始化函数作用与当前对象（self）
        super().__init__()                          ##调用初始化函数

        ##提取图片特征,利用Sequential模块组合
        self.features=nn.Sequential(
            ##第一层卷积
            nn.Conv2d(3,32,kernel_size=3,padding=1),##输入3通道，输出32通道，卷积核3×3，外围补一圈
            nn.ReLU(),                              ##ReLU激活函数，将负值变成0
            nn.MaxPool2d(2),                        ##最大池化，缩减宽高变为16×16，忽略边缘值
            ##第二层卷积
            nn.Conv2d(32,64,kernel_size=3,padding=1),##输入32通道，输出64通道
            nn.ReLU(),                              ##ReLU激活函数
            nn.MaxPool2d(2),                        ##最大池化，缩减宽高变为8×8
        )

        ##分类器，将特征转化为10类别,利用Sequential模块组合
        self.classifier=nn.Sequential(
            nn.Flatten(),                           ##将特征图转换成一维张量
            nn.Linear(64*8*8,128),                  ##全连接层，输出128个特征
            nn.ReLU(),                              ##ReLU激活函数，增加非线性
            nn.Linear(128,10),                      ##输出层，最后输出10个类别
        )
    
    ##定义图片处理顺序
    def forward(self,x):
        x=self.features(x)
        x=self.classifier(x)
        return x



# 3. 定义训练一轮函数
def train_one_epoch(model,train_loader,loss_fn,optimizer,device):
    model.train()                                   ##将模型转为训练模式

    total_loss=0.0                                  ##定义总损失,一定要用浮点数
    
    ##执行一个batch的处理
    for images,labels in train_loader:
        ##将图片和标签迁移至device处理              
        images=images.to(device)
        labels=labels.to(device)
        logits=model(images)                        ##利用model计算当前图片得分
        loss=loss_fn(logits,labels)                 ##利用输出得分和真实标签得分计算出损失值
        optimizer.zero_grad()                       ##清空上一轮梯度，防止积累
        loss.backward()                             ##反向传播计算梯度
        optimizer.step()                            ##更新参数
        total_loss+=loss.item()                     ##用loss.item()将张量转换成值并计入到总损失

    avg_loss=total_loss/len(train_loader)           ##计算平均损失

    return avg_loss




# 4. 定义评估函数
def evaluate(model,test_loader,loss_fn,device):
    model.eval()                                    ##将模型转为评估模式

    total_loss=0.0                                  ##定义总损失
    correct=0                                       ##定义样本正确数
    total=0                                         ##定义测试样本数

    with torch.no_grad():                           ##在无梯度下进行
        ##执行一个批次的样本
        for images,labels in test_loader:  
            ##将图片和标签迁移至device处理         
            images=images.to(device)
            labels=labels.to(device)
            logits=model(images)                    ##用model计算图片得分
            loss=loss_fn(logits,labels)             ##利用得分和真实类被得分计算损失
            total_loss+=loss.item()                 ##统计总损失
            predictions=logits.argmax(dim=1)        ##dim=1，对比每列找到最大值即为模型预测的类别
            correct+=(predictions==labels).sum().item()##统计预测正确的个数
            total+=labels.size(0)                   ##统计这个batch测试了多少个样本
    
    avg_loss=total_loss/len(test_loader)            ##计算平均损失
    accuracy=correct/total                          ##计算准确率

    return avg_loss,accuracy




# 5. 定义主函数与超参数
def main():
    #定义使用设备，cuda可用则使用cuda
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")

    #定义超参数,每个批次多少张图片,迭代轮次，学习率
    batch_size=64
    epochs=5
    learning_rate=0.01

    #利用transform处理图片
    transform=transforms.Compose([                  ## Compose组合处理图片
        transforms.ToTensor(),                      ## Totensor将rgb图片转换成张量并且像素值0~255缩小到0~1(C,H,W)
        transforms.Normalize((0.5,0.5,0.5),         ## Normalize定义三个值均为均值与方差0.5的正态0~1变-1~1
                             (0.5,0.5,0.5))
    ]
    )

    #使用torchvision的数据集接口下载并加载需要的CIFAR10训练集 
    train_dataset=torchvision.datasets.CIFAR10(
        root="./data",                              ##文件保存至同级目录的data
        train=True,                                 ##当前为训练集
        download=True,                              ##检查本地是否有，否则自动下载
        transform=transform,                        ##transform转换使用上面定义的transform
    )

    #使用torchvision的数据接口下载并加载需要的CIFAR10测试集
    test_dataset=torchvision.datasets.CIFAR10(
        root="./data",                              ##文件保存至同级目录的data      
        train=False,                                ##当前为测试集
        download=True,                              ##检查本地是否有，否则自动下载
        transform=transform,                        ##transform转换使用上面定义的transform
    )

    #使用DataLoader把数据分批次取出来
    ##训练数据集
    train_loader=DataLoader(
        train_dataset,                              ##读取之前创建的训练数据集
        batch_size=batch_size,                      ##每个批次读取定义的batch_size张图片及标签
        shuffle=True,                               ##打乱训练数据集，如若不打乱会影响模型的学习
        num_workers=0,                              ##采用0个并发进程执行
    )
    ##测试数据集
    test_loader=DataLoader(
        test_dataset,                               ##读取之前创建的测试数据集
        batch_size=batch_size,                      ##每个批次读取定义的batch_size张图片及标签
        shuffle=False,                              ##测试集无需打乱
        num_workers=0,                              ##采用0个并发进程执行
    )

    #创建模型并移动至gpu上运行
    model=SimpleCNN().to(device)                    ##实例化一个SimpleCNN模型对象送至device运行

    #采用交叉熵损失适用于多分类任务
    loss_fn=nn.CrossEntropyLoss()

    #创建sgd优化器
    optimizer=torch.optim.SGD(
        model.parameters(),                         ##将model里面的参数也放在优化器里面
        lr=learning_rate,                           ##定义优化器的学习率
        momentum=0.9,                               ##定义惯性量，往梯度下降的地方滑
    )

    #用列表记录每一轮训练loss，测试loss，测试准确率
    train_loss_history=[]
    test_loss_history=[]
    test_accuracy_history=[]

    #训练
    for epoch in range(epochs):
        
        ##定义训练损失，调用train_one_epoch函数执行一轮训练，并将所需要的参数传入该函数获得训练损失
        train_loss=train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
        )

        ##定义测试损失和准确率，调用evaluate函数评估这一轮训练结果，并将所需要的参数传入该函数获得测试损失和准确率
        test_loss,test_accuracy=evaluate(
            model,
            test_loader,
            loss_fn,
            device,
        )

        ##存储本轮结果，便于后续可视化
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        ##输出每轮文字结果
        print(
            f"当前轮次：{epoch+1}"
            f"   训练损失：{train_loss:.4f}"
            f"   测试损失：{test_loss:.4f}"
            f"   测试准确率：{test_accuracy*100:.2f}"
        )

    epoch_range=range(1,epochs+1)
    ##绘损失曲线
    plt.plot(epoch_range,train_loss_history,label="训练损失")
    plt.plot(epoch_range,test_loss_history,label="测试损失")
    plt.xlabel("轮次")
    plt.ylabel("损失值")
    plt.title("CIFAR10 损失曲线图")
    plt.legend()
    plt.grid(True)
    plt.show()

    ##绘准确率曲线
    plt.figure()
    plt.plot(epoch_range,test_accuracy_history,label="测试准确率")
    plt.xlabel("轮次")
    plt.ylabel("准确率")
    plt.title("CIFAR10 准确率曲线图")
    plt.legend()
    plt.grid(True)
    plt.show()

##执行程序入口
if __name__=="__main__":
    main()
