# 1. 导入库与全局参数
import torch
import torch.nn as nn
import torch.optim as optimizer
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
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
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DATASET_PATH=os.path.join(BASE_DIR,"data","cartpole_demo.npz")

# 2. expert_policy
def expert_policy(observation):
    pole_angle=float(observation[2])
    if pole_angle>0:
        return 1
    return 0

# 3. collect_one_episode
def collect_one_episdoe():
    env=gym.make(ENV_NAME)
    observation,info=env.reset()
    observations=[]
    actions=[]
    rewards=[]
    terminated=False
    truncated=False
    while not (terminated or truncated):
        action=expert_policy(observation)
        observations.append(observation)
        observation,reward,terminated,truncated,info=env.step(action)
        actions.append(action)
        rewards.append(reward)
    
    episode = {
    "observations": np.array(observations,dtype=np.float32),
    "actions": np.array(actions,dtype=np.int64),
    "rewards": np.array(rewards,dtype=np.float32),
    "terminated": terminated,
    "truncated": truncated,
    }
    env.close()
    return episode

# 4. collect_many_episodes
def collect_many_episodes(episode_count):
    episodes=[]
    for index in range(episode_count):
        episode=collect_one_episdoe()
        episodes.append(episode)
    
    return episodes

# 5. save_dataset
def save_dataset(episodes, dataset_path):
    os.makedirs(os.path.dirname(dataset_path),exist_ok=True)
    np.savez(dataset_path,episodes=np.array(episodes,dtype=object))

# 6. load_dataset
def load_dataset(dataset_path):
    data=np.load(dataset_path,allow_pickle=True)
    episodes=data["episodes"]
    return episodes

# 7. inspect_dataset
def inspect_dataset(episodes):
    print(f"episodes数量：{len(episodes)}")
    episode=episodes[0]
    print(f"observation shape:{episode['observations'].shape}")
    print(f"action shape:{episode['actions'].shape}")
    print(f"reward shape:{episode['rewards'].shape}")

# 8. main
def main():
    dataset_path=DATASET_PATH
    episode_count=5
    episodes=collect_many_episodes(episode_count)
    save_dataset(episodes,dataset_path)
    loaded_dataset=load_dataset(dataset_path)
    inspect_dataset(loaded_dataset)

# 9. 程序入口
if __name__=="__main__":
    main()
