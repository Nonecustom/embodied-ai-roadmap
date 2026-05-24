# Stage 05: ACT Policy

## Goal

从普通 behavior cloning 过渡到 ACT 的核心思想：一次不只预测一个 action，而是预测未来一段 action。

本阶段先不急着完整复现 ACT 论文，而是先理解最小变化：

```text
普通 BC:
observation[t] -> action[t]

Action Chunk:
observation[t] -> actions[t : t + K]
```

也就是：

```python
action_chunk = policy(observation)
```

## Why This Stage

Stage 04 已经完成：

```text
expert_policy
-> collect demonstration dataset
-> train behavior cloning policy
-> rollout learned policy
-> save/load model
```

但普通 BC 每次只预测一步 action。真实机械臂任务往往是连续动作过程，例如：

```text
靠近物体
-> 调整姿态
-> 闭合夹爪
-> 抬起
-> 移动到目标位置
-> 放下
```

所以 Stage 05 的第一个目标是理解：

```text
一个 observation 如何对应未来 K 步动作
```

## Repository Structure

```text
stage05-act-policy/
  README.md
  action_chunk_dataset.py
```

## Learning Logic Chain

```text
single-step BC
-> observation[t] -> action[t]
-> demonstration episode
-> action sequence
-> action chunk
-> observation[t] -> actions[t : t + K]
-> ACT
```

一句话理解：

```text
ACT 可以先理解为“把 BC 的单步动作标签，扩展成未来 K 步动作标签”。
```

## Key Concepts

- Action Chunk: 一段连续动作序列，例如未来 4 步动作。
- Chunk Size: 每个 action chunk 的长度，通常记为 `K`。
- Single-Step BC: 一个 observation 只预测一个 action。
- Sequence Prediction: 模型输出不再是单个动作，而是一段动作序列。
- Demonstration Episode: 一次完整专家示范轨迹，包含连续的 observations 和 actions。
- Training Sample: 用于训练模型的一条样本，例如 `observation[t]` 和 `actions[t:t+K]`。

## First Task

基于 Stage 04 保存的 CartPole demonstration dataset，构造 action chunk dataset。

输入：

```text
episodes
```

每条 episode 中有：

```text
observations: [T, 4]
actions: [T]
rewards: [T]
```

输出：

```text
chunk_observations: [N, 4]
action_chunks: [N, K]
```

其中：

```text
chunk_observations[i] = observation[t]
action_chunks[i] = actions[t : t + K]
```

## Results

当前使用：

```text
CHUNK_SIZE = 4
```

运行 `action_chunk_dataset.py` 后输出：

```text
chunk observations shape:(170, 4)
action chunks shape:(170, 4)
first observation:[-0.00597309  0.04341381 -0.04408776  0.03909043]
first action chunk:[0 0 0 0]
```

结果含义：

- 一共构造出 170 条 action chunk 训练样本。
- 每条 observation 是 CartPole 的 4 维状态。
- 每条 action chunk 包含未来 4 步动作。
- 第一条样本表示：在该 observation 下，专家未来 4 步动作都是 `0`。

当前数据格式已经从：

```text
observation[t] -> action[t]
```

升级为：

```text
observation[t] -> actions[t : t + K]
```

## My Understanding

1. action chunk 和普通 BC 的区别是什么？

```text
普通 BC 预测一步动作：observation[t] -> action[t]。
Action chunk 预测一段动作：observation[t] -> actions[t : t + K]。
```

2. 如果 episode 长度是 27，K=4，可以构造多少个 action chunk 样本？

```text
27 - 4 + 1 = 24
```

最后一个可用起点是：

```text
t = 23
```

对应：

```text
actions[23:27]
```

3. 为什么最后几个时间步不能直接构造完整 chunk？

```text
因为 label 是未来 K 步 action。如果 t 太靠后，后续 action 数量不够 K 个，就不能构造完整 action chunk。
```

例如：

```text
t = 24 -> actions[24:28] 只剩 3 个
t = 25 -> actions[25:29] 只剩 2 个
t = 26 -> actions[26:30] 只剩 1 个
```

4. 一次预测多个 action 对机械臂有什么好处？

```text
机械臂抓取任务通常由一系列连续动作组成，而不是某一个单独动作就能完成。Action chunk 能让模型一次预测一段动作，使动作更连续，也更接近真实操作过程。
```

## Completion Standard

- [x] 能加载 Stage 04 保存的 `cartpole_demo.npz`
- [x] 能理解一条 episode 中 `observations/actions` 的 shape
- [x] 能从一条 episode 构造 action chunk 样本
- [x] 能从多条 episode 构造完整 action chunk dataset
- [x] 能打印 `chunk_observations.shape` 和 `action_chunks.shape`
- [x] 能解释 `T - K + 1` 的来源
- [ ] 能训练一个最小 chunk policy
- [ ] 能解释 chunk policy 的输出 shape

## Guided Stage Review

### 1. 从 BC 到 ACT

普通 BC 的训练样本是：

```text
输入：observation[t]
标签：action[t]
```

Action chunk 的训练样本是：

```text
输入：observation[t]
标签：actions[t : t + K]
```

### 2. 数据变化

如果一条 episode 长度是 `T`，chunk size 是 `K`，可构造样本数是：

```text
T - K + 1
```

原因是：

```text
最后一个可用起点是 t = T - K，此时 actions[t:t+K] 仍然能取到完整 K 个动作。
```

### 3. 和机械臂项目的关系

真实机械臂中，一个 action chunk 可以表示：

```text
靠近物体、调整姿态、闭合夹爪、抬起等一小段连续动作计划。
```

### 4. 下一步

本阶段第一个代码目标：

```text
load demonstration dataset
-> build action chunk dataset
-> inspect chunk shapes
```

已完成。下一步是：

```text
chunk_observations
-> chunk policy
-> predicted action_chunks
-> loss(predicted_chunk, expert_chunk)
```
