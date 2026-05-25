# Stage 06: Diffusion Policy

## Goal

从 ACT / action chunk 继续推进到 Diffusion Policy，理解另一种生成动作序列的方法。

本阶段先不完整复现论文，而是先理解最小变化：

```text
ACT / Action Chunk:
observation -> action sequence

Diffusion Policy:
noise action sequence -> denoise -> action sequence
```

一句话理解：

```text
Diffusion Policy 不是直接预测动作序列，而是学习如何把带噪声的动作序列一步步还原成像专家一样的动作序列。
```

## Why This Stage

Stage 05 已经完成：

```text
demonstration dataset
-> action chunk dataset
-> chunk policy
-> predicted action chunks
-> temporal ensemble
-> final action sequence
```

但 chunk policy 本质上仍然是直接预测：

```text
observation -> actions[t:t+K]
```

Diffusion Policy 继续解决同一个问题：

```text
机械臂动作不是一个点，而是一段连续动作序列。
```

不同点在于：

```text
Action Chunk 直接输出动作序列
Diffusion Policy 通过去噪过程生成动作序列
```

## Learning Logic Chain

```text
控制任务需要输出连续动作序列
-> ACT / action chunk 直接预测未来 K 步动作
-> 直接预测可能难以表达多种可行动作分布
-> 引入 diffusion，把动作序列生成看成从噪声逐步还原
-> 模型学习 noisy action + timestep + observation -> noise
-> 推理时从随机噪声开始反复 denoise
-> 得到最终 action sequence
-> 为真实机械臂学习更平滑、更复杂的动作分布做准备
```

一句话说明：

```text
Stage 06 是从“直接预测动作序列”推进到“通过去噪生成动作序列”。
```

## Key Concepts

- Clean Action Sequence: 专家示范中的真实动作序列。
- Noise: 加到动作序列上的随机扰动。
- Noisy Action Sequence: 被噪声污染后的动作序列。
- Timestep: 扩散过程中的噪声等级。
- Denoise: 从 noisy action 中预测并去掉噪声。
- Conditional Generation: 根据 observation 条件生成动作序列。
- Receding Horizon Control: 生成一段动作，但每次只执行前面几步，然后重新观察再生成。

## First Task

第一轮不接机器人环境，只做 toy diffusion action demo。

目标：

```text
clean action sequence
-> add noise
-> model predicts noise
-> remove noise
-> generated action sequence
```

当前只需要理解三个问题：

```text
1. noise 是什么？
2. timestep 表示什么？
3. denoise 为什么能生成动作？
```

## Paper First Pass

论文笔记位于：

```text
E:\github clone repo\literature-notes\notes\embodied-intelligence\diffusion-policy.md
```

第一轮阅读只回答 8 个问题：

```text
1. 这篇论文解决什么问题？
2. 输入是什么？
3. 输出是什么？
4. 数据怎么来？
5. 模型结构是什么？
6. loss 是什么？
7. 实验怎么评估？
8. 我现在能复现哪一小部分？
```

## Planned Repository Structure

```text
stage06-diffusion-policy/
  README.md
  toy_diffusion_action.py
  cartpole_diffusion_policy.py
  images/
```

## Completion Standard

- [ ] 完成 Diffusion Policy 论文第一轮阅读
- [ ] 能解释 clean action / noisy action / predicted noise
- [ ] 能解释 timestep 的作用
- [ ] 跑通 toy diffusion action demo
- [ ] 画出 toy diffusion 的 loss 曲线
- [ ] 能从随机噪声生成一段简单动作序列
- [ ] 能说明 Diffusion Policy 和 ACT 的区别
- [ ] 决定是否接入 Stage 04/05 的 CartPole action chunk dataset

## Guided Stage Review

### 1. 本阶段最开始的问题是什么？

```text
ACT / action chunk 能直接预测一段动作，但真实机械臂任务中可能存在多种合理动作序列，直接回归或分类未必能很好表达动作分布。
```

### 2. 为了解决这个问题，引入了什么方法？

```text
引入 diffusion，把动作序列生成看成一个从噪声逐步还原的过程。
```

### 3. 这个方法又带来了什么新问题？

```text
需要理解 noise、timestep、denoise、sampling 这些新概念。
```

### 4. 下一步方法如何解决这个新问题？

```text
先写 toy diffusion action demo，不接视觉、不接机械臂，只让模型学习生成一段简单动作序列。
```

### 5. 本阶段最重要的数据流是什么？

```text
clean action sequence
-> add noise
-> noisy action sequence
-> predict noise
-> denoise
-> generated action sequence
```

### 6. 它和最终真实机械臂项目有什么关系？

```text
真实机械臂执行的是连续动作序列。Diffusion Policy 可以作为后续真实机械臂模仿学习策略的一种候选方法。
```

### 7. 当前还有哪些无法理解的点？

```text
待补充。
```
