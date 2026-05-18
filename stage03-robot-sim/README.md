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

- Environment:
- Observation:
- Action:
- Reward:
- Episode:
- Reset:
- Step:
- Terminated:
- Truncated:
- Policy:

## Results

| Experiment | Environment | Action Type | Output | Observation |
|---|---|---|---|---|
| Baseline |  | random action |  |  |
| Exp 1 |  |  |  |  |
| Exp 2 |  |  |  |  |

## Questions

1. `env.reset()` 的作用是什么？

2. observation 表示什么？

3. action 表示什么？

4. `env.step(action)` 返回了哪些内容？

5. reward 的作用是什么？

6. terminated 和 truncated 有什么区别？

7. random policy 为什么也有学习价值？

8. 仿真环境和真实机械臂之间有什么关系？

## Code Questions

这些问题来自 `*_reference.py` 中的 `#？` 标记，用来记录阅读参考代码时不理解或需要继续追问的地方。

1. 

## Reflection

This stage I learned:

1. 
2. 
3. 

## Completion Standard

- [ ] 能运行一个最小仿真环境
- [ ] 能说清楚 `reset` 和 `step` 的作用
- [ ] 能打印 observation 和 action
- [ ] 能写出 random policy
- [ ] 能记录一次完整 episode
- [ ] 能解释当前 demo 和后续机械臂控制的关系
