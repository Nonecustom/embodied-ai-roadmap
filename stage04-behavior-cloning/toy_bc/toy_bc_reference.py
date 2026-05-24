import torch
import torch.nn as nn
import torch.optim as optim


NUM_SAMPLES = 1000
OBS_DIM = 2
ACTION_DIM = 2
EPOCHS = 200
LEARNING_RATE = 0.01


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(                   #？梳理这个神经网络的步骤
            nn.Linear(OBS_DIM, 32),                 #？这里为什么要输出32维的隐藏特征
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, ACTION_DIM),
        )

    def forward(self, observation):
        action = self.net(observation)
        return action


def generate_observations(num_samples):
    observations = torch.randn(num_samples, OBS_DIM)
    return observations


def expert_policy(observations):
    x = observations[:, 0]
    y = observations[:, 1]
    action_1 = 2 * x                                #？为什么要这样计算动作维度
    action_2 = -1 * y
    expert_actions = torch.stack([action_1, action_2], dim=1)
    return expert_actions


def train_model(model, observations, expert_actions):
    loss_fn = nn.MSELoss()                          #？了解均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)        #？了解Adam这个优化器

    for epoch in range(EPOCHS):
        predicted_actions = model(observations)
        loss = loss_fn(predicted_actions, expert_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:                   #？为什么要每20轮打印一次，其他的数字呢
            print(f"epoch {epoch + 1}, loss: {loss.item():.6f}") 


def test_model(model):
    test_observation = torch.tensor([[1.0, 2.0]])   
    target_action = expert_policy(test_observation)

    with torch.no_grad():                           #？为什么在无梯度下进行(就一个测试样本，不存在损失计算梯度下降)
        predicted_action = model(test_observation)

    print(f"test observation: {test_observation}")
    print(f"expert action: {target_action}")
    print(f"predicted action: {predicted_action}")


def main():
    observations = generate_observations(NUM_SAMPLES)
    expert_actions = expert_policy(observations)

    model = PolicyNet()
    train_model(model, observations, expert_actions)
    test_model(model)


if __name__ == "__main__":
    main()
