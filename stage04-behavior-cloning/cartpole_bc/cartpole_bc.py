# 1. 导入库，定义超参数
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import io
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

ENV_NAME="CartPole-v1"
NUM_SAMPLES=2000
OBS_DIM=4
NUM_ACTIONS=2
EPOCHS=200
LEARNING_RATE=0.001

# 2. 定义策略网络 PolicyNet
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(OBS_DIM,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,NUM_ACTIONS)
        )

    def forward(self,observation):
        logits=self.net(observation)
        return logits

# 3. 定义专家策略 expert_policy
def expert_policy(observation):
    pole_angle=float(observation[2])            ##取observation第三个量：杆子角度
    if pole_angle>0:                            ##角度大于0，杆子右倾向右推保持平衡
        return 1
    return 0

# 4. 采集 observation/action 数据 collect_data
def collect_data(num_samples):
    env=gym.make(ENV_NAME)
    observation,info=env.reset()
    observations=[]
    actions=[]

    while len(observations)<num_samples:            ##采样数
        action=expert_policy(observation)           ##根据当前状态得出专家策略
        observations.append(observation)
        actions.append(action)

        observation,reward,terminated,truncated,info=env.step(action)      ##更新下一步

        if terminated or truncated:
            observation,info=env.reset()
    
    env.close() 
    observations=torch.tensor(np.array(observations),dtype=torch.float32)       ##模型处理tensor，因此需要转tensor
    actions=torch.tensor(np.array(actions),dtype=torch.long)
    return observations,actions


# 5. 训练模型 train_model
def train_model(model,observations,actions):
    loss_fn=nn.CrossEntropyLoss()
    epoch_loss=[]
    optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        logits=model(observations)
        loss=loss_fn(logits,actions)
        epoch_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    ##绘训练损失图
    epoch_range=range(1,EPOCHS+1)
    plt.plot(epoch_range,epoch_loss)
    plt.title("训练损失曲线图")
    plt.xlabel("训练轮次")
    plt.ylabel("损失值")
    plt.show()


# 6. 评估模型 evaluate_model
def evaluate_model(model,observations,actions):
    model.eval()

    with torch.no_grad():
        logits=model(observations)
        predicited_actions=logits.argmax(dim=1)
        accurcy=(predicited_actions==actions).float().mean().item()
    
    return accurcy

# 7. 单样本测试 test_model
def test_model(model):
    model.eval()

    test_observation=torch.tensor([[0.5,0.0,0.5,0.1]],dtype=torch.float32)
    expert_action=expert_policy(test_observation[0])

    with torch.no_grad():
        logits=model(test_observation)
        predicited_action=logits.argmax(dim=1).item()
    
    print(f"test observation:{test_observation}")
    print(f"expert action:{expert_action}")
    print(f"predicted action:{predicited_action}")

# 8. 主函数 main
def main():
    num_samples=NUM_SAMPLES
    observations,actions=collect_data(num_samples)
    print(f"observations shape:{observations.shape}")
    print(f"actions shape:{actions.shape}")

    model=PolicyNet()

    train_model(model,observations,actions)
    accuracy=evaluate_model(model,observations,actions)
    
    print(f"training accuracy:{accuracy:.4f}")
    test_model(model)

# 9. 程序入口
if __name__=="__main__":
    main()