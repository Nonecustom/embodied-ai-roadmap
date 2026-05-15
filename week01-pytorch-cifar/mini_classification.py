import torch
from torch import nn
import matplotlib.pyplot as plt
import sys
import io
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 1.构造数据集
x=torch.randn(100,4)  # 依据正态分布生成100×4的向量，现实意义为100个样本，每个样本有4个特征
y=torch.randint(0,3,(100,)) # 100个样本对应的标签，标签仅有三个类别，分别为0，1，2

# 2.定义模型】
model=nn.Linear(4,3) #模型输入为四个特征，输出为三个类别

# 3.定义损失函数
loss_f=nn.CrossEntropyLoss()  # 交叉熵损失作为损失函数 
loss_history=[] #损失列表存储每轮损失值便于绘图

# 4.定义优化器
optimizer=torch.optim.SGD(model.parameters(),lr=0.1) #优化器采用SGD梯度下降，学习率为0.1

# 5.训练循环
for epoch in range(50):
    goals=model(x) #用模型算出每个样本的得分
    loss=loss_f(goals,y)  #计算得分与真实类别之间的损失
    optimizer.zero_grad() #清空上一轮梯度
    loss.backward()  #反向传播计算梯度
    loss_history.append(loss.item())
    optimizer.step() #根据所得梯度值更新当前参数 
    print(f"第{epoch+1}轮loss:{loss.item():.4f}") #输出当轮损失值，以4位小数表示
    #   loss.item() 的作用是：把只有一个数的 tensor 转成 Python 普通数字。

plt.plot(loss_history)
plt.xlabel("迭代轮次")
plt.ylabel("Loss值")
plt.title("损失曲线")
plt.grid(True)
plt.show()
