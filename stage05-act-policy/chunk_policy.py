from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from action_chunk_dataset import build_chunk_dataset
from action_chunk_dataset import load_demonstration_dataset


# 1. 导入库与路径参数
BASE_DIR = Path(__file__).parent
REPO_DIR = BASE_DIR.parent
DATASET_PATH = (
    REPO_DIR
    / "stage04-behavior-cloning"
    / "cartpole_bc"
    / "data"
    / "cartpole_demo.npz"
)

OBS_DIM = 4
CHUNK_SIZE = 4
NUM_ACTIONS = 2
HIDDEN_DIM = 64
EPOCHS = 200
LEARNING_RATE = 0.001


# 2. 加载 action chunk dataset
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
        self.net=nn.Sequential(
            nn.Linear(OBS_DIM,HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM,HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM,CHUNK_SIZE*NUM_ACTIONS)
        )

    def forward(self, observations):
        logits=self.net(observations)
        logits=logits.reshape(-1,CHUNK_SIZE,NUM_ACTIONS)
        return logits


# 4. 训练 train_model
def train_model(model, observations, action_chunks):
    loss_fn=nn.CrossEntropyLoss()
    action_chunks=action_chunks.reshape(-1)
    episode_loss=[]
    optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE)
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
        



# 5. 单样本测试 test_model
def test_model(model, observation,action_chunks):
    model.eval()

    observation = observation.unsqueeze(0)

    with torch.no_grad():
        logits = model(observation)
        predicted_chunk = logits.argmax(dim=2)
        predicted_chunk = predicted_chunk.squeeze(0)

    print(f"test observation:{observation}")
    print(f"predicted action chunk:{predicted_chunk}")
    print(f"action chunk:{action_chunks}")
  
def evaluate_model(model, observations, action_chunks):
    logits = model(observations)
    predictions = logits.argmax(dim=2)

    element_accuracy = (predictions == action_chunks).float().mean().item()

    exact_accuracy = (predictions == action_chunks).all(dim=1).float().mean().item()
    print(f"element accuracy:{element_accuracy}")
    print(f"exact accuracy:{exact_accuracy}")

# 6. 主函数 main
def main():
    observations, action_chunks = load_action_chunk_dataset(
        DATASET_PATH,
        CHUNK_SIZE,
    )
    for i in range(len(action_chunks)):
        if (action_chunks[i] == 1).any():
            test_index = i
            break


    model = ChunkPolicyNet()
    train_model(model, observations, action_chunks)
    test_model(model, observations[test_index],action_chunks[test_index])
    evaluate_model(model,observations,action_chunks)


# 7. 程序入口
if __name__ == "__main__":
    main()
