# 具身智能项目驱动学习计划

> 版本日期：2026-05-17  
> 学习策略：项目驱动 + 最小必要理论 + 论文伴读 + 代码复现 + 实体闭环
> 当前状态：Stage 01 与 Stage 02 已完成，准备进入 Stage 03：机器人仿真。

## 1. 总目标

最终目标不是“看完很多课程”，而是做出一个可以展示的具身智能项目：

> 使用低成本桌面机械臂、摄像头和模仿学习算法，实现简单桌面物体抓取、分类和放置任务。

推荐最终项目：

```text
LeRobot / SO-101 机械臂
+ USB 摄像头
+ PyTorch 模仿学习算法
+ 桌面方块 / 盒子
= 机械臂根据视觉输入完成抓取和分类放置
```

目标展示形式：

- 一个 GitHub 仓库；
- 一段 2-3 分钟 demo 视频；
- 一份项目报告；
- 1-2 篇论文的最小复现或实验对比；
- 真实机械臂完成至少一个稳定任务。

复试展示定位：

> 最终项目要呈现为“低成本桌面机械臂模仿学习系统”，而不是泛泛地说“学习具身智能”。  
> 需要能讲清楚：问题定义、系统结构、数据采集、模型训练、实验结果、失败案例和改进方向。

## 2. 总体学习原则

不要采用“先学完所有基础再动手”的方式。

推荐比例：

| 内容 | 占比 |
|---|---:|
| 写代码 / 跑实验 | 60% |
| 读论文 / 读文档 | 25% |
| 看视频补基础 | 15% |

每个阶段都要问自己：

> 这个阶段我让系统多具备了什么能力？

而不是：

> 这个阶段我看完了多少视频？

新增学习规则：

- 看完关键视频后，生成一份对应的理解型 `.md` 笔记；
- 笔记不做视频摘抄，而是围绕“核心问题、名词解释、和代码的关系、和项目的关系、自测问题、我的理解”整理；
- API 不再堆在一个长文件里，而是按功能模块拆分，方便后续查找。
- 后续参考代码默认使用纯代码版本，不添加讲解型注释；学习时由我在 `*_reference.py` 中不理解的代码旁用 `#？` 标记问题，再把这些问题整理进对应阶段目录的 `README.md`。
- 手敲代码时，如果发现关键逻辑、易错点或重要理解，用 `#k` 标记；阶段整理时将这些内容归纳到对应 `README.md` 的 `Key Takeaways`。
- 每个阶段初期可以使用 `*_reference.py` 作为脚手架，帮助理解新概念和新库接口；但随着阶段推进，逐步减少直接给完整参考代码，改为先由我自己写目标拆解、伪代码、函数接口和最小实现。
- 遇到新任务时，优先训练独立代码能力：先回答“输入是什么、输出是什么、关键函数是什么、数据流怎么走、怎么验证是否成功”，再写代码；报错时先定位原因，再最小修改。
- 参考代码的作用是降低第一次接触的门槛，不是长期替代自己设计程序。阶段后半段必须安排 1-2 个“无完整 reference”的小任务，用来训练独立实现和 debug 能力。

阶段内代码能力推进方式：

```text
阶段初期：阅读 reference -> 标记 #？ -> 手敲复现
阶段中期：给目标和接口 -> 自己补实现 -> 报错后定位修正
阶段后期：自己拆任务 -> 写伪代码 -> 独立实现 -> 记录 debug log
```

## 3. 阶段路线总览

| 阶段 | 时间建议 | 核心目标 | 阶段产出 |
|---|---:|---|---|
| 阶段 1 | 1-2 周 | PyTorch 训练流程入门 | CIFAR10 分类代码、loss 曲线 |
| 阶段 2 | 1-2 周 | ResNet 图像特征 / CLIP 图文匹配 | 图像特征提取 demo、图文匹配 demo |
| 阶段 3 | 3-4 周 | 机器人仿真跑起来 | ManiSkill / MuJoCo 机械臂 demo |
| 阶段 4 | 3-4 周 | 模仿学习最小闭环 | BC baseline 训练与评估 |
| 阶段 5 | 4-6 周 | ACT / Diffusion Policy 论文伴读与复现 | 论文笔记、最小实验 |
| 阶段 6 | 6-8 周 | 真实机械臂项目 | 采集数据、训练策略、真实执行 |
| 阶段 7 | 2-3 周 | 项目整理与展示 | README、报告、视频、实验表 |

如果每天 1-2 小时学习，大约需要 5-8 个月跑通第一版。  
如果假期集中投入，可以压缩到 3-5 个月。

## 4. 阶段 1：PyTorch 与 CIFAR10 图像分类

建议时间：1-2 周。  
当前状态：已完成。

### 阶段目标

理解深度学习训练程序的基本结构：

```text
数据集
→ DataLoader
→ 模型
→ loss
→ optimizer
→ 训练循环
→ 测试评估
→ 画图记录
```

### 需要掌握的基础

- `torch.tensor`
- `torch.randn`
- `torch.randint`
- `requires_grad`
- `loss.backward()`
- `optimizer.step()`
- `nn.Linear`
- `nn.CrossEntropyLoss`
- `DataLoader`
- `matplotlib` 画 loss 曲线

### 任务清单

- [x] 整理 `notes/pytorch-api-notes.md`
- [x] 完成 mini classification 随机数据分类
- [x] 画出 mini classification 的 loss 曲线
- [x] 生成并理解 `train_reference.py`
- [x] 手敲 `train.py`
- [x] 跑通 CIFAR10 数据集
- [x] 记录训练 loss、测试 loss、测试准确率
- [x] 做至少 3 个实验：
  - [x] 改 epoch
  - [x] 改 learning rate
  - [x] 改 batch size
- [x] 在 README 中记录实验结果

### 推荐视频 / 资料

- 李沐《动手学深度学习》：
  - 数据操作
  - 自动求导
  - Softmax 回归
  - 多层感知机
  - LeNet / 卷积神经网络
  - ResNet
  - 数据增广
- PyTorch 官方教程：
  - 60-minute blitz
  - CIFAR10 classifier tutorial

### 阶段产出

```text
stage01-pytorch-cifar/
  mini_classification.py
  trainrefernence.py
  train.py
  mini_cf_images/
  CIFAR10_images/
  README.md
```

### 已完成总结

- 完成 PyTorch 基础小练习：张量操作、自动求导、拼接、矩阵乘法、loss 记录；
- 完成 mini classification 随机数据分类，并画出 loss 曲线；
- 完成 CIFAR10 CNN 分类代码手敲复现；
- 完成 baseline 与 3 组参数实验；
- 记录了 train loss、test loss、accuracy 和对应曲线；
- 理解了 `train_loss` 近似训练误差，`test_loss` 是泛化误差在测试集上的估计；
- 解决了 CUDA 版 PyTorch 安装问题，当前 GPU 可用：RTX 3050 Laptop GPU；
- 解决了 CIFAR10 数据集不应上传 GitHub 的问题，已将 `stage01-pytorch-cifar/data/` 加入 `.gitignore`。

## 5. 阶段 2：ResNet 图像特征与 CLIP 入门

建议时间：1-2 周。

当前状态：已完成。

### 阶段目标

从普通图像分类过渡到“机器人如何看懂图像”。

### 需要掌握的基础

- 图片张量形状：`[batch, channel, height, width]`
- CNN / ResNet 的特征提取作用
- embedding 的概念
- CLIP 的基本思想：图像和文本映射到同一个语义空间
- 训练技巧分支的轻量理解：过拟合、权重衰退、Dropout、数据增强、BatchNorm。

### 任务清单

- [x] 看完模型选择相关内容
- [x] 看完 ResNet 相关内容
- [x] 生成 `resnet-notes.md`
- [x] 使用预训练 ResNet18 提取一张图片的特征
- [x] 使用 CLIP 对图片和文本进行相似度匹配
- [x] 测试文本提示：
  - `a photo of a mouse`
  - `a photo of a keyboard`
  - `a photo of a cup`
  - `a photo of a book`
  - `a photo of a phone`
- [x] 写一个小 demo：输入图片，输出最相似的文本标签

### 推荐视频 / 资料

- 李沐：模型选择、ResNet、预训练模型、数据增广
- OpenAI CLIP 论文或项目介绍
- Hugging Face Transformers 入门文档

### 阶段产出

```text
stage02-vision-features/
  README.md
  resnet_feature_reference.py
  resnet_feature.py
  clip_match_reference.py
  clip_match.py
  images/
  results/

notes/
  vision/
    resnet-notes.md
```

### 本阶段理解重点

ResNet 不是本阶段最终目标，而是视觉编码器：

```text
图片
→ ResNet18
→ 图像特征向量
→ 后续分类 / 图文匹配 / 策略模型输入
```

后续机械臂项目中可能出现：

```python
image_feature = vision_encoder(image)
action = policy(image_feature, robot_state)
```

因此本阶段重点是理解“图片如何变成特征”，而不是从零实现 ResNet。

## 6. 阶段 3：机器人仿真跑起来

建议时间：3-4 周。

当前状态：已跑通 Gymnasium 与 MuJoCo 最小仿真闭环，已完成 random policy、多轮 episode 评估与函数化整理。

### 阶段目标

不要一开始系统学完整机器人学，先让机械臂在仿真中动起来。

### 需要掌握的基础

- observation：机器人看到或感知到的状态
- action：机器人要执行的动作
- episode：一次完整任务尝试
- joint：机械臂关节
- gripper：夹爪
- position control：位置控制
- camera：仿真相机
- reward / success：任务是否完成

### 任务清单

- [x] 先用 Gymnasium 跑通最小 `reset` / `step` 流程
- [x] 打印 observation、action、reward、terminated、truncated
- [x] 写一个 random policy 并运行完整 episode
- [x] 运行 5 个 episode，计算 average reward
- [x] 将 random policy 评估代码整理成函数结构
- [x] 再安装并跑通 MuJoCo
- [x] 跑通 MuJoCo 单步 `reset` / `step`
- [x] 跑通 MuJoCo random policy 完整 episode
- [x] 跑通 MuJoCo 5 个 episode，计算 average reward
- [x] 安装并跑通 ManiSkill
- [x] 运行一个简单机械臂环境
- [x] 打印 observation 和 action 的形状
- [x] 跑通 ManiSkill random policy 完整 episode
- [x] 跑通 ManiSkill 5 个 episode，计算 average reward
- [x] 跑通 ManiSkill render 可视化
- [x] 写一个 random policy
- [ ] 写一个简单控制脚本
- [ ] 让机械臂执行简单动作
- [ ] 记录成功和失败现象

### 卡住时再补的知识

| 卡住点 | 补什么 |
|---|---|
| 机械臂坐标不对 | 坐标系、位姿 |
| 关节动不了 | joint、action space |
| 末端到不了目标 | 正运动学、逆运动学 |
| 抓不住物体 | gripper、contact、控制频率 |

### 推荐资料

- ManiSkill 官方文档
- MuJoCo 官方文档
- MIT Robotic Manipulation 课程中与 manipulation 相关的章节
- 古月居 / 赵虚左 / 鱼香ROS 中的机器人基础内容

### 阶段产出

```text
stage03-robot-sim/
  run_env.py
  random_policy.py
  simple_control.py
  README.md
```

## 7. 阶段 4：模仿学习最小闭环

建议时间：3-4 周。

### 阶段目标

理解具身智能中的核心形式：

```text
observation → policy → action
```

也就是：

```python
action = policy(observation)
```

### 需要掌握的基础

- demonstration：专家或人类示范数据
- behavior cloning：行为克隆
- policy：策略模型
- success rate：任务成功率
- overfitting：过拟合
- generalization：泛化
- train / eval split：训练集和测试集划分

### 任务清单

- [x] 完成 toy behavior cloning 最小闭环
- [x] 画出 toy BC loss 曲线
- [x] 将 Stage 04 整理为 `toy_bc/` 与 `cartpole_bc/`
- [x] 跑通 CartPole behavior cloning reference
- [x] 手敲 CartPole behavior cloning
- [x] 记录 CartPole BC debug 过程和解决方法
- [x] 独立实现 CartPole learned policy rollout
- [x] 完成 learned policy 与 random policy 的 5 轮平均 reward 对比
- [x] 完成 CartPole BC 模型保存与加载，并使用加载后的模型 rollout
- [x] 构造 CartPole episode 格式 demonstration dataset，并完成保存、加载、检查
- [ ] 跑通 robomimic 或 LeRobot 中的 BC baseline
- [ ] 理解数据格式：
  - observation
  - action
  - state
  - image
  - episode
- [ ] 改数据量观察结果变化
- [ ] 改训练 epoch 观察结果变化
- [ ] 记录 success rate
- [ ] 写失败案例分析

### 推荐资料

- robomimic 文档
- LeRobot 文档
- Behavior Cloning 入门资料

### 阶段产出

```text
stage04-behavior-cloning/
  README.md
  toy_bc/
    toy_bc_reference.py
    toy_bc.py
    images/
  cartpole_bc/
    cartpole_bc_reference.py
    cartpole_bc.py
    cartpole_bc_rollout.py
    cartpole_bc_save_load.py
    cartpole_demonstration_dataset.py
    data/
```

## 8. 阶段 5：论文伴读与最小复现

建议时间：4-6 周。

### 阶段目标

不要纯读论文，每篇论文都要绑定代码问题。

读论文时只先回答 8 个问题：

1. 这篇论文解决什么问题？
2. 输入是什么？
3. 输出是什么？
4. 数据怎么来？
5. 模型结构是什么？
6. loss 是什么？
7. 实验怎么评估？
8. 我能复现哪一部分？

### 推荐论文顺序

| 顺序 | 论文 / 方向 | 重点 |
|---|---|---|
| 1 | ACT / Action Chunking Transformer | 一次预测多个动作 chunk |
| 2 | Diffusion Policy | 用扩散模型生成动作序列 |
| 3 | Mobile ALOHA | 低成本遥操作和真实任务设计 |
| 4 | Open X-Embodiment / RT-X | 大规模跨机器人数据 |
| 5 | RT-2 | 视觉语言模型到机器人动作 |
| 6 | Octo | 开源通用机器人策略 |
| 7 | OpenVLA | 开源 VLA 模型 |
| 8 | SmolVLA / pi0 | 更近期的高效 VLA / flow matching 路线 |

### 任务清单

- [x] 生成 `stage05-act-policy/README.md`
- [x] 理解普通 BC 与 action chunk 的区别
- [x] 从 CartPole demonstration dataset 构造 action chunk dataset
- [x] 打印并解释 `chunk_observations.shape` 和 `action_chunks.shape`
- [x] 完成 ACT 论文第一轮阅读：动机、输入输出、action chunking
- [x] 训练最小 chunk policy，理解模型输出 `[batch, K, num_actions]`
- [x] 计算 chunk policy 的 element accuracy 与 exact accuracy
- [x] 写 ACT 论文笔记
- [x] 实现 toy temporal ensemble demo：重叠 chunk 投票融合
- [x] 将 temporal ensemble 接到模型预测 action chunks 上
- [x] 生成 `stage06-diffusion-policy/README.md`
- [x] 生成 Diffusion Policy 论文笔记入口到 `literature-notes`
- [ ] 写 Diffusion Policy 论文笔记
- [ ] 跑 ACT 或 Diffusion Policy 的最小 demo
- [ ] 对比 BC 与 ACT / Diffusion Policy 的结果
- [ ] 总结这些方法如何迁移到自己的机械臂项目

### 论文笔记模板

```text
# Paper Name

## 解决什么问题

## 输入是什么

## 输出是什么

## 模型结构

## 数据怎么来

## loss 是什么

## 实验结果

## 我能复现哪一部分

## 对我的机械臂项目有什么用
```

### 阶段产出

```text
papers/
  ACT.md
  Diffusion-Policy.md
  Mobile-ALOHA.md
  OpenVLA.md
  SmolVLA.md

stage05-act-policy/
  README.md
  action_chunk_dataset.py
  chunk_policy.py
  experiments.md

stage06-diffusion-policy/
  README.md
  experiments.md
```

## 9. 阶段 6：真实机械臂项目

建议时间：6-8 周。

### 推荐硬件

| 材料 | 作用 | 估算价格 |
|---|---|---:|
| LeRobot SO-101 / SO-ARM101 机械臂 | 低成本真实机械臂实验 | 2500-4500 RMB |
| USB 摄像头 1-2 个 | 视觉输入 | 100-600 RMB |
| 桌面方块 / 盒子 / 小物体 | 操作任务道具 | 100-300 RMB |
| 摄像头支架 / 灯光 | 稳定视觉输入 | 100-300 RMB |
| 备用舵机 / 电源 / 线材 | 维护和稳定运行 | 300-800 RMB |
| Jetson Orin Nano | 可选，边缘部署 | 1800-2500 RMB |
| RealSense 深度相机 | 可选，深度视觉 | 1500-3000 RMB |

初期推荐预算：3500-6000 RMB。  
更舒适预算：6000-12000 RMB。

### 阶段目标

跑通真实闭环：

```text
摄像头采集图像
→ 读取机械臂状态
→ policy 输出动作
→ 机械臂执行
→ 再次观察
→ 循环
```

### 任务清单

- [ ] 组装机械臂
- [ ] 跑通官方 LeRobot 例程
- [ ] 测试遥操作
- [ ] 固定摄像头视角
- [ ] 采集 30 条固定位置抓取数据
- [ ] 训练第一个 BC / ACT 策略
- [ ] 测试真实机械臂成功率
- [ ] 扩充到 100-200 条数据
- [ ] 做红蓝方块分类放置任务
- [ ] 记录失败案例

### 推荐任务难度顺序

| 任务 | 难度 |
|---|---|
| 固定位置抓方块 | 入门 |
| 随机位置抓方块 | 基础 |
| 抓红色方块放左边盒子 | 推荐 |
| 红蓝方块分类放置 | 推荐最终任务 |
| 根据语言指令抓指定物体 | 进阶 |
| 杂乱桌面多物体抓取 | 较难 |

### 阶段产出

```text
real-robot/
  data-collection/
  policy-training/
  evaluation/
  demo-videos/
  README.md
```

## 10. 阶段 7：项目整理与展示

建议时间：2-3 周。

### 阶段目标

把学习成果整理成可以给别人看的项目，而不是散乱代码。

### GitHub 首页 README 应包含

- 项目目标
- 硬件清单
- 环境配置
- 数据采集方法
- 模型方法
- 实验结果
- 失败案例
- Demo 视频
- 论文参考
- 下一步计划

### 任务清单

- [ ] 整理仓库结构
- [ ] 补全 README
- [ ] 上传 loss / accuracy 曲线
- [ ] 整理论文笔记
- [ ] 录制 demo 视频
- [ ] 写项目总结报告
- [ ] 列出后续改进方向

## 11. 推荐仓库结构

```text
embodied-ai-roadmap/
  README.md
  requirements.txt
  environment.yml

  stage01-pytorch-cifar/
  stage02-vision-features/
  stage03-robot-sim/
  stage04-behavior-cloning/
    toy_bc/
    cartpole_bc/
  stage05-act-policy/
  stage06-diffusion-policy/

  papers/
    ACT.md
    Diffusion-Policy.md
    Mobile-ALOHA.md
    OpenVLA.md
    SmolVLA.md

  simulation/
    mujoco/
    maniskill/

  real-robot/
    lerobot-so101/
    data-collection/
    policy-training/
    evaluation/

  notes/
    pytorch-api-notes.md
    api/
      01-tensor-basic.md
      02-autograd.md
      03-data-loading.md
      04-model-layers.md
      05-training-loop.md
      06-evaluation-inference.md
      07-vision-transforms.md
      08-pretrained-vision-models.md
      09-plotting.md
      10-common-errors.md
    robotics-basics.md
    imitation-learning.md

  assets/
    images/
    videos/
```

## 12. 学习逻辑链与阶段复盘

每个阶段结束时，不只记录“做了什么”，还要整理“为什么这样学”。  
复盘重点是学习逻辑链，而不是写长篇总结。

固定复盘规则：

- 每个阶段 README 必须包含 `Learning Logic Chain` 和 `Guided Stage Review`；
- 逻辑链不再写成知识点堆叠，而是写成“问题 -> 方法 -> 新问题 -> 新方法”；
- 每个新知识点都要回答：它是为了解决上一个什么问题出现的；
- 每个阶段复盘都要说明：这个阶段如何把能力推进到最终机械臂项目；
- `Guided Stage Review` 用于记录阶段复盘、无法理解的点、关键提问和阶段性答案，不必让用户重复回答已经讲清楚的问题。

标准模板：

```text
## Learning Logic Chain

问题 1：当前任务遇到了什么限制？
-> 方法 1：引入什么概念 / 代码 / 模型来解决？
-> 新问题 2：方法 1 又带来了什么新问题？
-> 方法 2：继续引入什么机制解决？
-> 和最终机械臂项目的关系：这个阶段让系统多具备了什么能力？

一句话说明：
这个阶段从什么能力推进到了什么能力。

## Guided Stage Review

这个区域用于整理阶段复盘、关键问题、无法理解的点和阶段性答案。
如果问题已经在对话中讲清楚，可以直接整理答案，不必让我重复回答。

1. 本阶段最开始的问题是什么？

2. 为了解决这个问题，引入了什么方法？

3. 这个方法又带来了什么新问题？

4. 下一步方法如何解决这个新问题？

5. 本阶段最重要的代码接口 / 数据流是什么？

6. 它和最终真实机械臂项目有什么关系？

7. 当前还有哪些无法理解的点？

下一阶段最小目标：
- 
```

当前总逻辑链：

```text
mini classification
-> logits / scores
-> softmax probability
-> CIFAR10 image classification
-> CNN feature extraction
-> ResNet pretrained visual feature
-> CLIP image-text feature matching
-> Gymnasium env.reset / env.step
-> CartPole discrete action
-> random policy episode evaluation
-> MuJoCo continuous action
-> MuJoCo physics simulation and render
-> ManiSkill robot arm simulation
-> behavior cloning
-> learned policy rollout
-> random policy comparison
-> save/load policy checkpoint
-> demonstration dataset format
-> action chunk dataset
-> chunk policy
-> temporal ensemble
-> real robot policy
```

## 13. 笔记与 API 管理规范

### 视频 / 课程笔记

每看完一个关键视频或章节，生成一个对应的理解型 `.md` 文件。

推荐结构：

```text
学习目标
核心问题
关键名词解释
和代码的关系
和最终机械臂项目的关系
自测问题
我的理解
后续建议
```

当前已建立：

```text
notes/vision/resnet-notes.md
```

### API 手册

API 不再放在一个超长文件中，而是按功能模块拆分。

总入口：

```text
notes/pytorch-api-notes.md
```

模块目录：

```text
notes/api/
  01-tensor-basic.md
  02-autograd.md
  03-data-loading.md
  04-model-layers.md
  05-training-loop.md
  06-evaluation-inference.md
  07-vision-transforms.md
  08-pretrained-vision-models.md
  09-plotting.md
  10-common-errors.md
```

查找方式：

```text
张量操作 → 01
自动求导 → 02
数据加载 → 03
模型结构 → 04
训练循环 → 05
测试推理 → 06
图像预处理 → 07
ResNet / CLIP → 08
画图 → 09
报错 → 10
```

## 14. 当前最近任务

当前阶段：阶段 5，ACT / action chunk 最小理解。

最近目标：

- [x] 完成 Stage 01：PyTorch 与 CIFAR10 图像分类
- [x] 完成 Stage 02：ResNet 图像特征与 CLIP 图文匹配
- [x] 将仓库结构切换为 Stage 命名
- [x] 生成 `stage03-robot-sim/README.md`
- [x] 选择 Stage 03 的第一个仿真入口：Gymnasium
- [x] 跑通第一个 `env.reset()` 和 `env.step(action)`
- [x] 打印 observation、action、reward、terminated、truncated
- [x] 跑通 random policy 的完整 episode
- [x] 跑通 5 个 episode 并计算 average reward
- [x] 在 `stage03-robot-sim/README.md` 中记录实验结果和问题
- [x] 将 random policy 代码整理成函数：`run_one_episode` / `run_many_episodes` / `main`
- [x] 进入 MuJoCo 前，理解当前 CartPole demo 与机械臂仿真的对应关系
- [x] 跑通 MuJoCo 单步 step
- [x] 跑通 MuJoCo 10 步随机动作
- [x] 跑通 MuJoCo 完整 episode
- [x] 跑通 MuJoCo 5 个 episode 并计算 average reward
- [x] 尝试 MuJoCo render 可视化
- [x] 安装并跑通 ManiSkill
- [x] 跑通 ManiSkill PickCube-v1 单步 step
- [x] 跑通 ManiSkill 完整 episode
- [x] 跑通 ManiSkill 5 个 episode 并计算 average reward
- [x] 跑通 ManiSkill render 可视化
- [x] 整理 Stage 03 代码目录结构并提交 GitHub
- [x] 生成 `stage04-behavior-cloning/README.md`
- [x] 生成并理解 toy behavior cloning reference
- [x] 手敲 toy behavior cloning 代码
- [x] 跑通 observation -> policy -> action 的最小监督学习闭环
- [x] 画出 toy behavior cloning training loss 曲线
- [x] 在 Stage 04 README 中记录实验结果、`#？` 和 `#k`
- [x] 整理 Stage 04 目录结构：`toy_bc/` 与 `cartpole_bc/`
- [x] 生成 CartPole behavior cloning reference
- [x] 手敲并跑通 CartPole behavior cloning
- [x] 在 Stage 04 README 中记录 CartPole BC 的 debug 过程
- [x] 让训练出的 CartPole policy 回到环境中运行一个 episode
- [x] 比较 random policy 和 behavior cloning policy 的区别
- [x] 记录 learned policy 与 random policy 的平均 reward 对比：46.0 vs 23.8
- [x] 总结 behavior cloning 的局限：训练准确率高不等于 rollout 一定稳定
- [x] 保存并加载 CartPole BC 模型，加载后 rollout 结果：reward=54.0, steps=54
- [x] 进入真实 demonstration 数据格式理解：episode / observation / action / state / image
- [x] 保存并检查 CartPole demonstration dataset：episodes=5, first_episode_len=27
- [x] 决定下一阶段入口：进入 ACT 的 action chunk 最小理解
- [x] 生成 `stage05-act-policy/README.md`
- [x] 生成 `stage05-act-policy/action_chunk_dataset.py` 半脚手架
- [x] 完成 `action_chunk_dataset.py`：从 episode 构造 `observation[t] -> actions[t:t+K]`
- [x] action chunk dataset 输出：`chunk_observations=(170, 4)`, `action_chunks=(170, 4)`
- [x] 更新 ACT 论文笔记到 `literature-notes`
- [x] 生成 `chunk_policy.py` 半脚手架
- [x] 完成最小 chunk policy 训练：`observation -> action_chunk`
- [x] chunk policy 结果：element accuracy=0.975, exact accuracy=0.90
- [x] 理解 ACT temporal ensemble：多个重叠 action chunk 如何融合成最终动作
- [x] 实现 toy temporal ensemble demo：重叠 chunk 投票融合，输出 `[0, 0, 1, 1, 0, 0]`
- [x] 将 temporal ensemble 接到模型预测 action chunks 上，输出 `[0, 0, 0, 0, 0, 1]`
- [ ] 收束 Stage 05：检查代码结构、整理结果、记录未理解问题
- [x] 决定下一阶段入口：Diffusion Policy 论文
- [x] 生成 `stage06-diffusion-policy/README.md`
- [x] 生成 `literature-notes/notes/embodied-intelligence/diffusion-policy.md`
- [ ] 第一轮阅读 Diffusion Policy，只回答 8 个问题
- [ ] 理解 noise / timestep / denoise 三个概念
- [ ] 设计 toy diffusion action demo 的输入、输出和验证方式

当前不要做：

- 不要系统刷完全部深度学习课程；
- 不要提前啃复杂 VLA 论文；
- 不要过早购买昂贵机械臂；
- 不要在 ROS2 上花太多无目标时间。
- 不要在训练技巧分支上过度停留，权重衰退、Dropout、BatchNorm 先做到理解用途即可。

当前最重要的是：

> 进入 Diffusion Policy 第一轮：先理解它为什么从“直接预测动作序列”变成“通过去噪生成动作序列”。
