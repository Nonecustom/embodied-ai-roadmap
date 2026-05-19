# Stage 03: Robot Simulation

## Goal

理解机器人仿真环境的最小运行流程，先跑通一个可以 `reset` 和 `step` 的环境。

本阶段重点不是系统学习完整机器人学，而是先理解：

```text
observation -> action -> env.step(action) -> next observation
```

## What I Will Do

- 安装并跑通一个最小仿真环境
- 创建环境 `env`
- 调用 `env.reset()` 获取初始 observation
- 构造一个 action
- 调用 `env.step(action)` 让环境前进一步
- 打印 observation、action、reward、terminated、truncated
- 写一个 random policy demo
- 记录仿真运行中的报错和疑问

## Recommended Order

1. 先用 Gymnasium 理解最小 `env.step()` 流程
2. 再进入 MuJoCo 或 ManiSkill
3. 最后再跑机械臂相关环境

## Key Concepts

- Environment: 仿真环境，负责保存当前状态，并根据动作返回新的状态和奖励。
- Observation: 环境返回给程序的观测信息，在 CartPole 中是长度为 4 的数组。
- Action: 程序给环境的动作，在 CartPole 中 `0` 表示向左施加力，`1` 表示向右施加力。
- Reward: 环境对当前动作结果的评分，CartPole 中每坚持一步通常得到 `1.0`。
- Episode: 从 `reset` 开始，到 `terminated` 或 `truncated` 为 True 结束的一整轮尝试。
- Reset: 重置环境，开始一轮新的 episode。
- Step: 执行动作，让环境前进一步。
- Terminated: 任务自然结束，例如杆子倒下或状态越界。
- Truncated: 因为时间步数限制等外部条件导致 episode 结束。
- Policy: 根据 observation 选择 action 的规则或模型，当前使用 random policy。

## Results

| Experiment | Environment | Action Type | Output | Observation |
|---|---|---|---|---|
| Baseline | CartPole-v1 | random action | step once, reward=1.0 | 成功跑通 `reset` 和一次 `step` |
| Exp 1 | CartPole-v1 | random action | total_reward=14.0, total_episode=14 | 成功运行一个完整 episode |
| Exp 2 | CartPole-v1 | random action | average_reward=15.0 over 5 episodes | 成功完成多轮随机策略评估 |
| Exp 3 | CartPole-v1 | random action | average_reward=21.8 over 5 episodes | 将 random policy 整理成函数结构并验证通过 |

## Questions

1. `env.reset()` 的作用是什么？
    重置当前环境状态
2. observation 表示什么？
    表示观测信息，在 CartPole 中是一个长度为 4 的数组，分别表示小车位置、小车速度、杆子角度和杆子角速度。

3. action 表示什么？
    表示动作，在 CartPole 中是一个离散值，`0` 表示向左施加力，`1` 表示向右施加力。

4. `env.step(action)` 返回了哪些内容？
    返回了 `next_observation`（执行动作后的新观测）、`reward`（这一步动作获得的奖励）、`terminated`（任务是否自然结束）、`truncated`（是否因为时间限制等外部条件结束）和 `info`（环境返回的额外调试信息）。

5. reward 的作用是什么？
    当前动作的奖励评分

6. terminated 和 truncated 有什么区别？
    `terminated` 表示任务自然结束，例如杆子倒下或状态越界；`truncated` 表示因为时间步数限制等外部条件导致 episode 结束。

7. random policy 为什么也有学习价值？
    random policy 可以作为一个基线，帮助我们理解环境的基本交互流程，并评估后续更复杂策略的改进效果。通过观察 random policy 的表现，我们可以了解环境的难度和奖励结构，为后续设计更智能的策略提供参考。
    
8. 仿真环境和真实机械臂之间有什么关系？
   `CartPole 中：`
    observation -> 小车和杆子的状态
    action -> 左/右施加力
    policy -> 根据状态选择动作
    reward -> 杆子保持平衡的评分

    `机械臂中：`
    observation -> 机械臂关节状态 + 摄像头图像
    action -> 关节角度/末端位姿/夹爪开合
    policy -> 根据视觉和状态选择动作
    reward/success -> 是否抓取成功

## Code Questions

这些问题来自 `gym_env_reference.py` 中的 `#？` 标记，用来记录阅读参考代码时不理解或需要继续追问的地方。

1. `ENV_NAME = "CartPole-v1"` 这个强化学习环境的作用是什么？
    `CartPole-v1` 是 Gymnasium 中的经典入门环境。任务目标是通过向左或向右给小车施加力，让杆子尽可能长时间保持不倒。它的作用不是模拟真实机械臂，而是用最小成本理解强化学习和机器人仿真的共同接口：`reset`、`step`、observation、action、reward 和 episode。

2. `reset_env(env)` 的作用是什么？为什么不先 `reset` 就会报 `Cannot call env.step() before calling env.reset()`？
    `reset_env(env)` 用来开始一轮新的 episode，并获得初始 observation。Gymnasium 要求环境必须先初始化状态，才能执行动作。如果没有调用 `env.reset()`，环境还没有初始状态，所以直接 `env.step(action)` 会报错。可以理解为：`reset` 是开局，`step` 是行动；没有开局，不能行动。

3. `next_observation, reward, terminated, truncated, info = env.step(action)` 中各个变量是什么意思？
    `next_observation` 是执行动作后的新观测；`reward` 是这一步动作获得的奖励；`terminated` 表示任务是否自然结束，例如杆子倒下；`truncated` 表示是否因为时间限制等外部条件结束；`info` 是环境返回的额外调试信息。后续机器人仿真中也会围绕这些信息判断策略表现。

4. `reward` 的作用是什么，后续有什么用？
    `reward` 是环境对当前动作结果的评分。CartPole 中每坚持一步通常得到 `1.0`，所以总奖励可以表示这一轮坚持了多少步。后续在强化学习或机器人策略评估中，reward 可以用来衡量策略好坏，也可以作为训练算法优化策略的目标。

## Reflection

This stage I learned:

1. 理解了 Gymnasium 仿真的最小闭环：`reset` 获取初始 observation，`step(action)` 执行动作并返回新状态。
2. 理解了 CartPole 的 observation space 是长度为 4 的数组，action space 是 `Discrete(2)`。
3. 学会了用 random policy 跑完整 episode，并通过多轮 episode 的平均奖励评估策略表现。
4. 学会了把一段脚本整理成 `run_one_episode`、`run_many_episodes` 和 `main`，让实验逻辑更清晰。

## Completion Standard

- [x] 能运行一个最小仿真环境
- [x] 能说清楚 `reset` 和 `step` 的作用
- [x] 能打印 observation 和 action
- [x] 能写出 random policy
- [x] 能记录一次完整 episode
- [x] 能将 random policy 评估代码整理成函数结构
- [ ] 能解释当前 demo 和后续机械臂控制的关系
