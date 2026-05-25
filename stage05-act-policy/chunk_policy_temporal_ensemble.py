# 1. 导入库与路径参数
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from action_chunk_dataset import build_chunk_dataset
from action_chunk_dataset import load_demonstration_dataset

CHUNK_SIZE=4
NUM_ACTIONS=2
OBS_DIM=4
EPOCHS = 200
LEARNING_RATE = 0.001

BASE_DIR=Path(__file__).parent
REPO_DIR=BASE_DIR.parent
DATASET_PATH=(
    REPO_DIR
    / "stage04-behavior-cloning"
    / "cartpole_bc"
    / "data"
    / "cartpole_demo.npz"
)

# 2. 复用 / 定义数据加载函数load_action_chunk_dataset
def load_action_chunk_dataset(dataset_path, chunk_size):
    episodes=load_demonstration_dataset(dataset_path)
    observations,action_chunks=build_chunk_dataset(episodes,chunk_size)
    observations=torch.tensor(observations,dtype=torch.float32)
    action_chunks=torch.tensor(action_chunks,dtype=torch.long)
    return observations,action_chunks

# 3. 定义 ChunkPolicyNet
class ChunkPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, CHUNK_SIZE * NUM_ACTIONS),
        )

    def forward(self, observations):
        logits = self.net(observations)
        logits = logits.reshape(-1, CHUNK_SIZE, NUM_ACTIONS)
        return logits

# 4. 训练模型train_model
def train_model(model,observations,action_chunks):
    loss_fn=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE)
    episode_loss=[]
    action_chunks=action_chunks.reshape(-1)
    for epoch in range(EPOCHS):
        logits=model(observations)
        logits=logits.reshape(-1,NUM_ACTIONS)
        loss=loss_fn(logits,action_chunks)
        episode_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    epoch_range=range(1,EPOCHS+1)
    plt.plot(epoch_range,episode_loss)
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.show()
    

# 5. 用模型预测多个 action chunk，predict_action_chunks
def predict_action_chunks(model,observations,start_index,chunk_count):
    model.eval()
    selected_observations=observations[start_index:start_index+chunk_count]
    with torch.no_grad():
        logits=model(selected_observations)
        selected_actions=logits.argmax(dim=2)
        selected_actions=selected_actions.tolist()

    return selected_actions

# 6. 收集重叠预测collect_votes
def collect_votes(action_chunks):
    votes={}                                  ##用字典表示，每个时间会有多个动作预测，因此字典有每个时间的list
    for chunk_index in range(len(action_chunks)):
        action_chunk=action_chunks[chunk_index]         ##读出每个chunk
        for action_index in range(len(action_chunk)):   ##遍历每个chunk的预测的每个action
            global_time=chunk_index+action_index        ##计算出属于预测总时间的哪个时间
            action=action_chunk[action_index]           ##读出当前动作预测

            if global_time not in votes:                ##当votes里面没有这个时间的list创建一个
                votes[global_time]=[]

            votes[global_time].append(action)

    return votes

# 7. 投票融合最终动作vote_actions
def vote_actions(votes):
    final_actions=[]
    for index in range(len(votes)):
        candidate_actions=votes[index]
        predicted_0=0
        predicted_1=0
        for action_index in range(len(candidate_actions)):
            if candidate_actions[action_index]==0:
                predicted_0+=1
            else:
                predicted_1+=1
        if predicted_0>=predicted_1:
            final_actions.append(0)
        else:
            final_actions.append(1)

    return final_actions

# 8. 打印结果inspect_results
def inspect_results(action_chunks, votes, final_actions):
    print("action chunks:")
    for chunk_index in range(len(action_chunks)):
        print(f"chunk {chunk_index}: {action_chunks[chunk_index]}")

    print("\nvotes:")
    for time_step in range(len(votes)):
        print(f"time {time_step}: {votes[time_step]}")

    print("\nfinal actions:")
    print(final_actions)

# 9. 主函数 main
def main():
    dataset_path=DATASET_PATH
    observations,action_chunks=load_action_chunk_dataset(dataset_path,CHUNK_SIZE)
    model=ChunkPolicyNet()
    train_model(model,observations,action_chunks)
    start_index=0
    chunk_count=3
    predicted_action_chunks=predict_action_chunks(model,observations,start_index,chunk_count)
    votes=collect_votes(predicted_action_chunks)
    final_actions=vote_actions(votes)
    inspect_results(predicted_action_chunks,votes,final_actions)

# 10. 程序入口
if __name__=="__main__":
    main()