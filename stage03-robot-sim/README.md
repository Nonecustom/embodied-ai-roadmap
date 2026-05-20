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

## Learning Logic Chain

```text
CartPole-v1
-> env.reset / env.step
-> observation / action / reward / terminated / truncated
-> discrete action: Discrete(2)
-> random policy episode
-> multi-episode average reward
-> function-based evaluation code
-> MuJoCo InvertedPendulum-v5
-> continuous action: Box(-3.0, 3.0)
-> physics simulation
-> render visualization
-> ManiSkill robot arm simulation
```

This stage connects basic environment interaction to continuous-control robot simulation.

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
- MuJoCo: 物理仿真引擎，用来模拟重力、关节、速度、力、碰撞等物理过程。
- Continuous Action: 连续动作，动作不是固定的 `0/1`，而是在某个范围内取浮点数，例如 `[-3.0, 3.0]`。
- Render: 可视化渲染，把仿真状态显示成窗口动画，帮助理解 action 对物理系统的影响。
- ManiSkill: 面向机器人操作任务的仿真平台，本阶段使用 `PickCube-v1` 进入机械臂任务。
- Batched Observation: 带 batch 维度的观测，例如 ManiSkill 中的 observation space 是 `(1, 42)`，表示 1 个并行环境和 42 维状态。

## Results

| Experiment | Environment | Action Type | Output | Observation |
|---|---|---|---|---|
| Baseline | CartPole-v1 | random action | step once, reward=1.0 | 成功跑通 `reset` 和一次 `step` |
| Exp 1 | CartPole-v1 | random action | total_reward=14.0, total_episode=14 | 成功运行一个完整 episode |
| Exp 2 | CartPole-v1 | random action | average_reward=15.0 over 5 episodes | 成功完成多轮随机策略评估 |
| Exp 3 | CartPole-v1 | random action | average_reward=21.8 over 5 episodes | 将 random policy 整理成函数结构并验证通过 |
| Exp 4 | InvertedPendulum-v5 | continuous random action | step once, reward=1 | 成功跑通 MuJoCo 单步物理仿真 |
| Exp 5 | InvertedPendulum-v5 | continuous random action | terminated=True at step 4 | 随机连续动作较大，倒立摆很快失稳 |
| Exp 6 | InvertedPendulum-v5 | continuous random action | total_reward=2.0, step_count=3 | 成功运行 MuJoCo 完整 episode |
| Exp 7 | InvertedPendulum-v5 | continuous random action | average_reward=5.6 over 5 episodes | 完成 MuJoCo 多轮随机策略评估 |
| Exp 8 | InvertedPendulum-v5 | continuous random action | render_mode="human" | 成功弹出可视化窗口，并通过 `time.sleep(0.03)` 放慢显示速度 |
| Exp 9 | PickCube-v1 | 8D continuous random action | step once, reward=0.0413 | 成功跑通 ManiSkill 机械臂任务单步交互 |
| Exp 10 | PickCube-v1 | 8D continuous random action | total_reward=1.9347, steps=50, truncated=True | 成功运行 ManiSkill 完整 episode |
| Exp 11 | PickCube-v1 | 8D continuous random action | average_reward=2.9505 over 5 episodes | 完成 ManiSkill 多轮随机策略评估 |
| Exp 12 | PickCube-v1 | 8D continuous random action | render_mode="human" | 成功弹出 SAPIEN 可视化窗口，需主动调用 `env.render()` 刷新画面 |

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

9. CartPole 和 MuJoCo InvertedPendulum 有什么区别？
    CartPole 的动作空间是 `Discrete(2)`，动作只能是向左或向右施加力。MuJoCo InvertedPendulum 的动作空间是 `Box(-3.0, 3.0, (1,))`，动作是连续控制量。连续动作更接近机械臂控制，因为机械臂关节角度、关节速度、末端位姿变化通常都是连续值。

10. MuJoCo 底层物理仿真的意义是什么？
    MuJoCo 的 `step(action)` 不只是改变数组，而是在物理引擎中根据力、速度、关节、重力等因素模拟一小段时间后的状态变化。后续机械臂仿真中，夹爪接触、物体滑动、关节运动、重力影响都需要这类物理仿真支持。

## Code Questions

这些问题来自 `*_reference.py` 中的 `#？` 标记，用来记录阅读参考代码时不理解或需要继续追问的地方。

### CartPole Reference

1. `ENV_NAME = "CartPole-v1"` 这个强化学习环境的作用是什么？
    `CartPole-v1` 是 Gymnasium 中的经典入门环境。任务目标是通过向左或向右给小车施加力，让杆子尽可能长时间保持不倒。它的作用不是模拟真实机械臂，而是用最小成本理解强化学习和机器人仿真的共同接口：`reset`、`step`、observation、action、reward 和 episode。

2. `reset_env(env)` 的作用是什么？为什么不先 `reset` 就会报 `Cannot call env.step() before calling env.reset()`？
    `reset_env(env)` 用来开始一轮新的 episode，并获得初始 observation。Gymnasium 要求环境必须先初始化状态，才能执行动作。如果没有调用 `env.reset()`，环境还没有初始状态，所以直接 `env.step(action)` 会报错。可以理解为：`reset` 是开局，`step` 是行动；没有开局，不能行动。

3. `next_observation, reward, terminated, truncated, info = env.step(action)` 中各个变量是什么意思？
    `next_observation` 是执行动作后的新观测；`reward` 是这一步动作获得的奖励；`terminated` 表示任务是否自然结束，例如杆子倒下；`truncated` 表示是否因为时间限制等外部条件结束；`info` 是环境返回的额外调试信息。后续机器人仿真中也会围绕这些信息判断策略表现。

4. `reward` 的作用是什么，后续有什么用？
    `reward` 是环境对当前动作结果的评分。CartPole 中每坚持一步通常得到 `1.0`，所以总奖励可以表示这一轮坚持了多少步。后续在强化学习或机器人策略评估中，reward 可以用来衡量策略好坏，也可以作为训练算法优化策略的目标。

### MuJoCo Reference

1. `ENV_NAME = "InvertedPendulum-v5"` 这个环境和 `CartPole-v1` 有什么异同？
    二者都是倒立摆类任务，目标都是让杆子尽可能保持平衡，并且都使用 `reset` / `step` / observation / action / reward / terminated / truncated 这套 Gymnasium 接口。区别在于：`CartPole-v1` 是入门级控制环境，动作空间是 `Discrete(2)`，只能选择向左或向右施加力；`InvertedPendulum-v5` 是 MuJoCo 物理仿真环境，动作空间是 `Box(-3.0, 3.0, (1,))`，动作是连续控制量。后者更接近机械臂控制，因为真实机器人通常需要输出连续的关节角度、速度或力。

2. 为什么 `mujoco_render.py` 中 `range(200)` 太小，而且窗口很快结束？
    MuJoCo 的仿真循环运行速度很快，如果不人为暂停，200 步会在很短时间内执行完。同时随机动作容易让倒立摆快速失败，如果失败后直接 `break`，窗口会很快关闭。解决方式是增加步数，例如 `range(1000)`，在每一步后加入 `time.sleep(0.03)` 放慢显示速度，并在 `terminated` 或 `truncated` 时重新 `reset`，这样窗口可以持续显示。

### ManiSkill Reference

1. `gym.make(ENV_NAME, obs_mode="state", render_mode=None)` 这几个参数的作用是什么？
    `ENV_NAME` 指定要创建的任务环境，本阶段使用 `PickCube-v1`。`obs_mode="state"` 表示使用低维状态向量作为 observation，而不是摄像头图片；本次输出的 observation space 是 `(1, 42)`，表示 1 个并行环境和 42 维状态。`render_mode=None` 表示不打开可视化窗口，只在后台运行，适合快速测试和训练。

## Key Takeaways

这些关键点来自手敲代码中的 `#k` 标记。

1. `reset_env(env)` 必须放在每个 episode 循环内部。每轮 episode 都需要重新初始化环境状态，否则上一轮的 `terminated` 或 `truncated` 会影响下一轮。

2. ManiSkill 的 `terminated` 和 `truncated` 是 `torch.Tensor`，所以初始化和判断时要注意类型；判断结束通常使用 `terminated.item()` 和 `truncated.item()`。

3. `render_mode="human"` 只是设置显示模式，真正刷新窗口画面需要主动调用 `env.render()`；如果循环太快，还需要用 `time.sleep()` 让画面可见。

## Reflection

This stage I learned:

1. 理解了 Gymnasium 仿真的最小闭环：`reset` 获取初始 observation，`step(action)` 执行动作并返回新状态。
2. 理解了 CartPole 的 observation space 是长度为 4 的数组，action space 是 `Discrete(2)`。
3. 学会了用 random policy 跑完整 episode，并通过多轮 episode 的平均奖励评估策略表现。
4. 学会了把一段脚本整理成 `run_one_episode`、`run_many_episodes` 和 `main`，让实验逻辑更清晰。
5. 理解了 MuJoCo 环境和 CartPole 一样使用 `reset` / `step` 接口，但动作空间从离散动作变成了连续控制量。
6. 学会了使用 `render_mode="human"` 打开 MuJoCo 可视化窗口，并用 `time.sleep()` 控制动画速度。
7. 理解了 ManiSkill 的 PickCube-v1 是机械臂操作任务，observation 变成 `(1, 42)` 状态向量，action 变成 8 维连续控制。
8. 学会了打开 ManiSkill / SAPIEN 可视化窗口，并理解 `render_mode`、`env.render()`、`time.sleep()` 的配合关系。

## Guided Stage Review

这个区域用于阶段结束时自己复盘，不要求一开始就写完。先用填空和问题把逻辑讲清楚。

### 1. 学习顺序

本阶段我的学习顺序是：

```text
CartPole
-> ________
-> ________
-> MuJoCo InvertedPendulum
-> ________
-> ________
-> ManiSkill PickCube
-> ________
-> ________
```

我为什么不直接从 ManiSkill 开始？

```text

```

### 2. CartPole 到 MuJoCo

CartPole 和 MuJoCo InvertedPendulum 的共同点是：

```text

```

它们最关键的区别是：

```text
CartPole 的 action space 是：________
MuJoCo 的 action space 是：________
这个区别说明：________
```

### 3. 核心接口

本阶段最重要的仿真接口是：

```python
observation, info = env.reset()
next_observation, reward, terminated, truncated, info = env.step(action)
```

我对这两行代码的理解：

```text
env.reset() 的作用是：________
env.step(action) 的作用是：________
reward 表示：________
terminated 表示：________
truncated 表示：________
```

### 4. 和机械臂项目的关系

CartPole 中：

```text
observation -> ________
action -> ________
reward -> ________
```

机械臂中：

```text
observation -> ________
action -> ________
reward / success -> ________
```

### 5. 下一阶段最小目标

进入 ManiSkill 前，我需要确认自己已经能做到：

```text
[ ] 能解释 reset / step
[ ] 能解释 observation / action
[ ] 能解释离散动作和连续动作的区别
[ ] 能运行 random policy
[ ] 能用 average reward 记录多轮表现
```

ManiSkill 的第一个最小目标是：

```text

```

## Completion Standard

- [x] 能运行一个最小仿真环境
- [x] 能说清楚 `reset` 和 `step` 的作用
- [x] 能打印 observation 和 action
- [x] 能写出 random policy
- [x] 能记录一次完整 episode
- [x] 能将 random policy 评估代码整理成函数结构
- [x] 能解释当前 demo 和后续机械臂控制的关系
- [x] 能跑通 MuJoCo 单步 step
- [x] 能跑通 MuJoCo 完整 episode 和多轮 average reward
- [x] 能打开 MuJoCo render 可视化窗口
- [x] 能跑通 ManiSkill PickCube-v1 单步 step
- [x] 能跑通 ManiSkill 完整 episode 和多轮 average reward
- [x] 能打开 ManiSkill render 可视化窗口
