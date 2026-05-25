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
  chunk_policy.py
  temporal_ensemble_demo.py
  chunk_policy_temporal_ensemble.py
```

## File Summary

```text
action_chunk_dataset.py
作用：读取 Stage 04 保存的 demonstration dataset，并构造 action chunk dataset。
核心数据流：episodes -> observation[t] -> actions[t:t+K]

chunk_policy.py
作用：使用 action chunk 思路训练一个最小策略模型。
核心数据流：observation -> predicted action chunk

temporal_ensemble_demo.py
作用：用手写 toy action chunks 演示 temporal ensemble。
核心问题：当多个 chunk 都预测同一个时间步的动作时，如何融合成最终动作。

chunk_policy_temporal_ensemble.py
作用：把 temporal ensemble 接到模型预测出来的 action chunks 上。
核心数据流：observation -> model -> predicted action chunks -> votes -> final actions
```

## Learning Logic Chain

```text
控制任务不是一次分类，而是一段连续动作过程
-> 普通 BC 学习 observation[t] -> action[t]
-> 单步动作预测容易抖动，也容易把误差带到后续状态
-> 引入 action chunk，让模型学习 observation[t] -> actions[t : t + K]
-> 模型不再只学“下一步怎么动”，而是学“一小段动作计划”
-> 每个时间步都会预测未来 K 步，导致同一个未来时间步可能收到多个预测
-> 引入 temporal ensemble，把重叠预测融合成最终执行动作
-> ACT 可以先理解为 action chunking + temporal ensemble 的策略学习方法
```

一句话理解：

```text
Stage 05 是从“单步模仿动作”升级到“预测一段动作计划”，再学习如何把重叠动作计划融合成稳定控制动作。
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

## Chunk Policy Result

文件：

```text
chunk_policy.py
```

当前最小模型：

```text
输入: observations [batch, 4]
输出: logits [batch, 4, 2]
标签: action_chunks [batch, 4]
```

单样本测试结果：

```text
test observation:tensor([[-0.0081, -0.3455, -0.0370,  0.5963]])
predicted action chunk:tensor([0, 0, 0, 1])
action chunk:tensor([0, 0, 0, 1])
```

整体评估结果：

```text
element accuracy:0.9750000238418579
exact accuracy:0.8999999761581421
```

两个指标的区别：

- Element accuracy：把所有 chunk 展平成单个动作，逐动作计算准确率。
- Exact accuracy：一个 chunk 内 4 个动作必须全部预测正确，才算这个 chunk 正确。

因此：

```text
element accuracy 高，不代表 exact accuracy 一定同样高。
```

因为一个 action chunk 只要错一个动作，整个 chunk 就不能算完全预测正确。

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

## ACT Paper First Pass

论文笔记位于：

```text
E:\github clone repo\literature-notes\notes\embodied-intelligence\act-action-chunking-transformer.md
```

第一轮阅读只关注 ACT 的动机和 action chunking，不要求理解完整 Transformer / CVAE / temporal ensemble。

已回答的问题：

```text
1. 论文解决低成本硬件上的精细双臂操作问题。
2. 普通 BC 单步预测容易动作抖动和误差累积。
3. action chunking 是一次预测未来 K 步动作。
4. ACT 输入包括图像观测和机器人关节状态等。
5. ACT 输出未来 K 步 action chunk。
6. Transformer 用于建模观测和动作序列之间的关系。
7. 低成本 leader-follower 系统用于采集 demonstration。
8. 当前能复现 action chunk dataset 构造。
```

## Temporal Ensemble

Temporal ensemble 要解决的问题：

```text
每个时间步都会预测一个未来 action chunk。
如果 chunk 之间重叠，同一个未来时间步的 action 可能会被多个不同起点的 chunk 预测到。
因此需要把这些重叠预测融合成最终执行动作。
```

例子：

```text
t = 0 预测 actions[0:4] -> 包含 action[2]
t = 1 预测 actions[1:5] -> 包含 action[2]
t = 2 预测 actions[2:6] -> 包含 action[2]
```

所以：

```text
action[2] 会有多个预测结果
```

融合方式：

- 连续动作：可以平均或指数加权平均。
- 离散动作：可以用投票，或平均 logits 后再取 `argmax`。

阶段理解：

```text
action chunk 解决“一次预测一段动作”
temporal ensemble 解决“多段重叠预测怎么合成最终动作”
```

## Toy Temporal Ensemble Demo

文件：

```text
temporal_ensemble_demo.py
```

当前 toy action chunks：

```text
chunk 0: [0, 0, 1, 1]
chunk 1: [0, 1, 1, 1]
chunk 2: [1, 1, 0, 0]
```

它们对应的全局时间步关系是：

```text
chunk 0 -> time 0, 1, 2, 3
chunk 1 -> time 1, 2, 3, 4
chunk 2 -> time 2, 3, 4, 5
```

因此同一个时间步会收到多个预测：

```text
time 0: [0]
time 1: [0, 0]
time 2: [1, 1, 1]
time 3: [1, 1, 1]
time 4: [1, 0]
time 5: [0]
```

投票融合后得到最终动作序列：

```text
[0, 0, 1, 1, 0, 0]
```

其中 `time 4: [1, 0]` 是平票，当前 demo 里暂时规定平票选 `0`。

这个 toy demo 对应 ACT 里的最小 temporal ensemble 思想：

```text
多个重叠 action chunk
-> 收集同一个全局时间步的多个预测
-> 融合成一个最终动作
```

## Model Temporal Ensemble Demo

文件：

```text
chunk_policy_temporal_ensemble.py
```

这一步把 toy demo 中手写的 action chunks，替换成模型预测出来的 action chunks：

```text
observations
-> trained chunk_policy
-> predicted action_chunks
-> collect_votes
-> vote_actions
-> final_actions
```

当前输出：

```text
action chunks:
chunk 0: [0, 0, 0, 0]
chunk 1: [0, 0, 0, 0]
chunk 2: [0, 0, 0, 1]

votes:
time 0: [0]
time 1: [0, 0]
time 2: [0, 0, 0]
time 3: [0, 0, 0]
time 4: [0, 0]
time 5: [1]

final actions:
[0, 0, 0, 0, 0, 1]
```

这说明当前已经完成从：

```text
手写 action_chunks -> temporal ensemble
```

升级到：

```text
模型预测 action_chunks -> temporal ensemble
```

其中 `final_actions` 的长度是：

```text
chunk_count + CHUNK_SIZE - 1
```

例如当前：

```text
3 + 4 - 1 = 6
```
形状链：

```text
episode observations: [T, 4]
episode actions: [T]

-> build chunk dataset

chunk_observations: [N, 4]
action_chunks: [N, 4]

-> model

raw output: [N, 8]
logits: [N, 4, 2]

-> loss reshape

logits: [N*4, 2]
labels: [N*4]

-> predict 3 chunks

selected_observations: [3, 4]
predicted_logits: [3, 4, 2]
predicted_action_chunks: [3, 4]

-> temporal ensemble

votes: 每个全局时间步对应若干预测
final_actions: [3 + 4 - 1] = [6]
```

## Completion Standard

- [x] 能加载 Stage 04 保存的 `cartpole_demo.npz`
- [x] 能理解一条 episode 中 `observations/actions` 的 shape
- [x] 能从一条 episode 构造 action chunk 样本
- [x] 能从多条 episode 构造完整 action chunk dataset
- [x] 能打印 `chunk_observations.shape` 和 `action_chunks.shape`
- [x] 能解释 `T - K + 1` 的来源
- [x] 完成 ACT 论文第一轮阅读问题
- [x] 能训练一个最小 chunk policy
- [x] 能解释 chunk policy 的输出 shape
- [x] 能计算 element accuracy 和 exact accuracy
- [x] 能用 toy demo 理解 temporal ensemble 的重叠预测与投票融合
- [x] 能将 temporal ensemble 接到模型预测的 action chunks 上

## Guided Stage Review

### 1. 从控制问题到普通 BC

CartPole 或机械臂控制不是普通分类问题，因为：

```text
分类任务通常只需要输出一个类别；控制任务需要连续输出动作，并且当前动作会影响下一时刻的状态。
```

普通 BC 先把控制问题简化成：

```text
observation[t] -> action[t]
```

它的意义是：

```text
模仿专家在当前状态下会采取什么动作。
```

### 2. 从普通 BC 到 Action Chunk

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

这一步的动机是：

```text
真实操作通常不是一个动作完成的，而是一小段连续动作完成的。
```

例如机械臂抓取可以拆成：

```text
靠近物体 -> 调整姿态 -> 闭合夹爪 -> 抬起
```

### 3. 数据变化

如果一条 episode 长度是 `T`，chunk size 是 `K`，可构造样本数是：

```text
T - K + 1
```

原因是：

```text
最后一个可用起点是 t = T - K，此时 actions[t:t+K] 仍然能取到完整 K 个动作。
```

### 4. 从 Action Chunk 到 Temporal Ensemble

Action chunk 带来的新问题是：

```text
每个时间步都会预测未来 K 步动作，因此同一个未来时间步可能会被多个 chunk 重复预测。
```

例如：

```text
chunk 0 预测 time 0, 1, 2, 3
chunk 1 预测 time 1, 2, 3, 4
chunk 2 预测 time 2, 3, 4, 5
```

所以：

```text
time 2 和 time 3 都会收到多个预测。
```

Temporal ensemble 的作用是：

```text
把同一个时间步的多个预测融合成一个最终执行动作。
```

在 CartPole 离散动作里，可以用：

```text
投票
```

在真实机械臂连续动作里，更常用：

```text
平均 / 指数加权平均
```

### 5. 和机械臂项目的关系

真实机械臂中，一个 action chunk 可以表示：

```text
靠近物体、调整姿态、闭合夹爪、抬起等一小段连续动作计划。
```

Temporal ensemble 则可以帮助：

```text
减少单次预测抖动，让连续控制动作更平滑。
```

### 6. 已完成代码链

当前 Stage 05 已经完成：

```text
load demonstration dataset
-> build action chunk dataset
-> inspect chunk shapes
-> chunk policy
-> predicted action_chunks
-> loss(predicted_chunk, expert_chunk)
-> evaluate element accuracy / exact accuracy
-> toy temporal ensemble demo
-> overlapping chunks
-> collect votes by global time
-> vote final actions
-> model predicted action chunks
-> model temporal ensemble
```

### 7. 下一步

```text
已完成：
模型预测 action_chunks -> collect_votes -> vote_actions
```

这表示 Stage 05 的最小 ACT 核心链条已经跑通：

```text
demonstration dataset
-> action chunk dataset
-> chunk policy
-> predicted action chunks
-> temporal ensemble
-> final action sequence
```

关键接口仍然是：

```text
observations: [batch, 4]
action_chunks: [batch, 4]
logits: [batch, 4, 2]
```

下一步不是继续堆代码，而是做一次阶段收束：

```text
1. 检查 Stage 05 代码结构
2. 整理当前所有输出结果
3. 补充无法理解的点
4. 决定下一阶段先进入 Diffusion Policy 论文，还是先做 LeRobot / SO-ARM101 准备
```
