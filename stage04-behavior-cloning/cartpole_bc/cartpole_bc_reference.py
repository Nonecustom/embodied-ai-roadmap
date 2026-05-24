import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


ENV_NAME = "CartPole-v1"
NUM_SAMPLES = 2000
OBS_DIM = 4
NUM_ACTIONS = 2
EPOCHS = 200
LEARNING_RATE = 0.001


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_ACTIONS),
        )

    def forward(self, observation):
        logits = self.net(observation)
        return logits


def expert_policy(observation):
    pole_angle = float(observation[2])              #？observation各个量的物理意义
    if pole_angle > 0:
        return 1
    return 0


def collect_data(num_samples):
    env = gym.make(ENV_NAME)
    observation, info = env.reset()
    observations = []
    actions = []

    while len(observations) < num_samples:
        action = expert_policy(observation)
        observations.append(observation)
        actions.append(action)

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

    observations = torch.tensor(np.array(observations), dtype=torch.float32)#？这两步的作用和原因，为什么要先转numpy这步多余吗
    actions = torch.tensor(actions, dtype=torch.long)
    return observations, actions


def train_model(model, observations, actions):
    loss_fn = nn.CrossEntropyLoss()                         #？了解交叉熵和均方误差的适用规则
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        logits = model(observations)
        loss = loss_fn(logits, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch + 1} loss:{loss.item():.6f}")


def evaluate_model(model, observations, actions):
    model.eval()
    with torch.no_grad():
        logits = model(observations)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == actions).float().mean().item()#？解读代码
    return accuracy


def test_model(model):
    test_observation = torch.tensor([[0.0, 0.0, 0.1, 0.0]], dtype=torch.float32)
    expert_action = expert_policy(test_observation[0])

    model.eval()
    with torch.no_grad():
        logits = model(test_observation)
        predicted_action = logits.argmax(dim=1).item()

    print(f"test observation:{test_observation}")
    print(f"expert action:{expert_action}")
    print(f"predicted action:{predicted_action}")


def main():
    observations, actions = collect_data(NUM_SAMPLES)
    model = PolicyNet()

    print(f"observations shape:{observations.shape}")
    print(f"actions shape:{actions.shape}")

    train_model(model, observations, actions)
    accuracy = evaluate_model(model, observations, actions)

    print(f"training accuracy:{accuracy:.4f}")
    test_model(model)


if __name__ == "__main__":
    main()
