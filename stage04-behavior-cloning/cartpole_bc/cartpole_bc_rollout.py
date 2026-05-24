# 1. 导入库与全局参数
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import sys
import io
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

ENV_NAME="CartPole-v1"
NUM_SAMPLES=2000
LEARNING_RATE=0.001
OBS_DIM=4
ACTION_DIM=2
EPOCHS=200

# 2. 复用 / 定义 PolicyNet
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(OBS_DIM,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,ACTION_DIM)
        )
    
    def forward(self,observation):
        logits=self.net(observation)
        return logits

# 3. 复用 expert_policy 与 collect_data
def expert_policy(observation):
    pole_angle=float(observation[2])
    if pole_angle>0:
        return 1
    return 0

def collect_data(num_samples):
    env=gym.make(ENV_NAME)
    observation,info=env.reset()
    observations=[]
    actions=[]

    while len(observations)<num_samples:            
        action=expert_policy(observation)           
        observations.append(observation)
        actions.append(action)

        observation,reward,terminated,truncated,info=env.step(action)      

        if terminated or truncated:
            observation,info=env.reset()
    
    env.close() 
    observations=torch.tensor(np.array(observations),dtype=torch.float32)       ##模型处理tensor，因此需要转tensor
    actions=torch.tensor(np.array(actions),dtype=torch.long)
    return observations,actions

# 4. 训练模型 train_model
def train_model(model,observations,actions):
    loss_fn=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE)
    epoch_loss=[]

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

# 5. observation 转 tensor
def observation_to_tensor(observations):
    observations=torch.tensor(observations)
    observations_tensor=observations.unsqueeze(0)
    return observations_tensor

# 6. 根据模型选择动作 select_action
def select_action(model,observations):
    model.eval()
    observations_tensor=observation_to_tensor(observations)
    with torch.no_grad():
        logits=model(observations_tensor)
        actions=logits.argmax(dim=1).item()
    return actions

# 7. 运行一个完整 episode run_one_episode
def run_one_episode(model):
    total_reward=0.0
    step_count=0
    env=gym.make(ENV_NAME)
    observation,info=env.reset()
    terminated=False
    truncated=False
    while not (terminated or truncated):
        action=select_action(model,observation)
        observation,reward,terminated,truncated,info=env.step(action)
        total_reward+=reward
        step_count+=1
    
    env.close()

    return total_reward,step_count

# 8. 运行多轮学习动作
def run_many_episodes(model, episode_count):
    episode_reward=[]
    episode_steps=[]
    for episode in range(episode_count):
        total_reward,step_count=run_one_episode(model)
        episode_reward.append(total_reward)
        episode_steps.append(step_count)
    
    average_reward=sum(episode_reward)/len(episode_reward)
    average_steps=sum(episode_steps)/len(episode_steps)
    return average_reward,average_steps


# 9. 运行一轮随机动作
def run_random_episode():
    env=gym.make(ENV_NAME)
    observation,info=env.reset()
    terminated=False
    truncated=False
    total_reward=0.0
    step_count=0
    while not (terminated or truncated):
        action=env.action_space.sample()
        next_observation,reward,terminated,truncated,info=env.step(action)
        total_reward+=reward
        step_count+=1
    
    env.close()
    return total_reward,step_count



# 10. 运行多轮随机动作
def run_many_random_episodes(episode_count):
    episode_reward=[]
    episode_steps=[]
    for epoch in range(episode_count):
        total_reward,step_count=run_random_episode()
        episode_reward.append(total_reward)
        episode_steps.append(step_count)

    average_reward=sum(episode_reward)/len(episode_reward)
    average_steps=sum(episode_steps)/len(episode_steps)
    return average_reward,average_steps


#   主函数 main及入口
def main():
    num_samples=NUM_SAMPLES
    episode_count=5
    model=PolicyNet()
    observations,actions=collect_data(num_samples)
    train_model(model,observations,actions)
    learned_average_reward,learned_average_steps=run_many_episodes(model,episode_count)
    random_average_reward,random_average_steps=run_many_random_episodes(episode_count)

    print(f"learned average reward:{learned_average_reward}")
    print(f"learned average steps:{learned_average_steps}")
    print(f"random average reward:{random_average_reward}")
    print(f"random average steps:{random_average_steps}")

if __name__=="__main__":
    main()