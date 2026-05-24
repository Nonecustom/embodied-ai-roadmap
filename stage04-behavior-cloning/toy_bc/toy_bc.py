# 1. 导入库
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import io
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号


# 2. 定义超参数
NUM_SAMPLES=1000                        ##样本数量
OBS_DIM=2                               ##观测维度为2维
ACTION_DIM=2                            ##动作维度也为2维
EPOCHS=200                              ##训练轮次
LEARNING_RATE=0.01                      ##学习率

# 3. 定义 PolicyNet
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(OBS_DIM,32),      ##第一层全连接层
            nn.ReLU(),                  ##激活函数
            nn.Linear(32,32),           ##第二层全连接层
            nn.ReLU(),                  ##激活函数
            nn.Linear(32,ACTION_DIM)    ##输出层
        )
    
    def forward(self,observation):      ##前向传播函数由当前状态observation输出动作
        action=self.net(observation)    ##调用自身网络
        return action

# 4. 生成 observations
def generate_observations(num_samples):
    observations=torch.randn(num_samples,OBS_DIM)
    return observations

# 5. 定义 expert_policy
def expert_policy(observations):
    x=observations[:,0]
    y=observations[:,1]
    action_1=2*x
    action_2=-1*y
    expert_actions=torch.stack([action_1,action_2],dim=1)
    return expert_actions

# 6. 训练模型 train_model
def train_model(model,observations,expert_actions):
    loss_fn=nn.MSELoss()                        ##使用均方误差作为损失函数
    optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE)       ##使用Adam优化器
    epoch_loss=[]

    for epoch in range(EPOCHS):                 ##开始训练
        predicted_actions=model(observations)   ##利用网络计算预测动作
        loss=loss_fn(predicted_actions,expert_actions)  ##计算训练损失

        optimizer.zero_grad()                   ##清空上一轮梯度
        loss.backward()                         ##反向传播计算梯度
        optimizer.step()                        ##依据梯度更新参数
        epoch_loss.append(loss.item())
    
    epoch_range=range(1,EPOCHS+1)
    plt.plot(epoch_range,epoch_loss)
    plt.title("训练损失曲线图")
    plt.xlabel("训练轮次")
    plt.ylabel("损失值")
    plt.show()

# 7. 测试模型 test_model
def test_model(model):
    test_observation=torch.tensor([[0.2,0.1]]) #k 默认batchsize为1了
    target_action=expert_policy(test_observation) ##计算专家动作

    with torch.no_grad():
        predicted_action=model(test_observation)  ##预测动作
    
    ##打印输出
    print(f"test observation: {test_observation}")
    print(f"expert action: {target_action}")
    print(f"predicted action: {predicted_action}")


# 8. main() 串起流程
def main():
    model=PolicyNet()
    observations=generate_observations(NUM_SAMPLES)
    expert_actions = expert_policy(observations)

    train_model(model,observations,expert_actions)
    test_model(model)

if __name__=="__main__":
    main()