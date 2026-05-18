# Softmax 回归学习笔记

## 1. 学习目标

看完 Softmax 回归后，我需要理解的不是完整推导所有公式，而是：

```text
分类任务为什么需要输出多个类别分数；
softmax 如何把 logits 转换成概率分布；
交叉熵损失如何衡量分类预测错误；
为什么 PyTorch 的 CrossEntropyLoss 输入的是 logits。
```

本阶段重点：

- 理解分类任务和回归任务的区别；
- 理解 logits、softmax、概率分布的关系；
- 理解交叉熵损失的直观含义；
- 理解 Stage 01 中 `nn.CrossEntropyLoss()` 的输入输出。

## 2. Softmax 回归想解决什么问题？

线性回归通常输出一个连续数值，例如房价、温度、分数。

分类任务需要输出一个类别，例如：

```text
猫 / 狗 / 鸟
飞机 / 汽车 / 船
0 / 1 / 2 / ... / 9
```

因此，分类模型通常不是只输出一个数，而是输出多个类别对应的分数。

例如 3 分类模型可能输出：

```text
[2.0, 1.0, 0.1]
```

这些数本身不是概率，而是模型对每个类别的原始打分，通常叫：

```text
logits
```

Softmax 的作用就是把这些分数变成类似概率的分布。

## 3. 核心概念：logits

`logits` 是模型最后一层直接输出的类别分数。

例如：

```python
logits = model(X)
```

如果是 3 分类，输出形状可能是：

```text
[batch_size, 3]
```

其中每一行表示一个样本属于 3 个类别的原始分数。

注意：

```text
logits 不是概率；
logits 可以是负数；
logits 的和不一定等于 1。
```

## 4. Softmax 是什么？

Softmax 把一组类别分数转换成概率分布。

公式：

```text
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

它有三个特点：

```text
每个输出都是正数；
所有输出加起来等于 1；
原始分数越大，对应概率通常越大。
```

例如：

```text
logits = [2.0, 1.0, 0.1]
softmax 后约为 [0.66, 0.24, 0.10]
```

这可以理解为模型认为：

```text
第 0 类概率最大；
第 1 类其次；
第 2 类最小。
```

## 5. 为什么不用普通除法？

如果直接用普通除法把分数除以总和，会遇到两个问题：

```text
模型输出可能有负数；
总和可能为 0 或者导致结果不稳定。
```

Softmax 先使用指数函数：

```text
exp(x)
```

把所有数变成正数，再除以指数和，从而得到合法的概率分布。

## 6. 交叉熵损失是什么？

分类训练时，我们希望模型给真实类别更高的概率。

如果真实类别对应概率是 `q`，交叉熵损失可以直观理解为：

```text
loss = -log(q)
```

当模型对真实类别很有把握时：

```text
q 接近 1
-log(q) 接近 0
loss 小
```

当模型对真实类别预测很低时：

```text
q 接近 0
-log(q) 很大
loss 大
```

因此，交叉熵损失会鼓励模型提高真实类别的预测概率。

## 7. PyTorch 中为什么不用手动 softmax？

在 PyTorch 中，多分类任务通常写：

```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
```

这里输入的是：

```text
logits: [batch_size, num_classes]
labels: [batch_size]
```

注意：

```text
CrossEntropyLoss 内部已经包含 softmax 相关计算；
所以传入 CrossEntropyLoss 前，不需要手动 softmax。
```

不要写成：

```python
prob = softmax(logits)
loss = loss_fn(prob, labels)
```

Stage 01 中正确写法是：

```python
logits = model(images)
loss = loss_fn(logits, labels)
```

## 8. Softmax 回归和 Stage 01 代码的关系

Stage 01 的 mini classification：

```python
X = torch.randn(100, 4)
y = torch.randint(0, 3, (100,))

model = nn.Linear(4, 3)
loss_fn = nn.CrossEntropyLoss()
```

含义：

```text
100 个样本；
每个样本 4 个特征；
一共 3 个类别；
模型输出 3 个 logits；
CrossEntropyLoss 根据 logits 和真实标签计算 loss。
```

Stage 01 的 CIFAR10：

```python
logits = model(images)
loss = loss_fn(logits, labels)
```

含义：

```text
每张图片输出 10 个类别分数；
labels 是 0 到 9 的类别编号；
loss 衡量模型对真实类别预测得好不好。
```

## 9. 关键名词解释

### Classification

分类任务，把输入样本分到某个类别中。

### Logits

模型最后一层直接输出的原始类别分数，不是概率。

### Softmax

把 logits 转换成概率分布的函数。

### Probability Distribution

概率分布，每个值非负，所有值加起来等于 1。

### Cross Entropy

衡量预测概率分布和真实类别之间差距的损失函数。

### Label

真实类别编号。例如 CIFAR10 中标签是 `0~9`。

## 10. 自测问题

1. 分类任务和回归任务有什么区别？
    分类任务输出的是类别或类别概率，回归任务输出的是连续数值。

2. 模型最后为什么要输出多个数？
    因为每个数对应一个类别的原始分数，多分类任务需要比较不同类别的可能性。

3. logits 和 probability 有什么区别？
    logits 是原始分数，可以为负，和不一定为 1；probability 是概率，非负且总和为 1。

4. softmax 的作用是什么？
    softmax 把一组 logits 转换成概率分布，使得每个类别都有对应概率。

5. 为什么交叉熵损失可以写成 `-log(q)`？
    因为 `q` 是真实类别对应的预测概率，`q` 越接近 1，`-log(q)` 越小；`q` 越接近 0，损失越大。

6. PyTorch 中 `CrossEntropyLoss` 的输入应该是什么？
    输入应该是模型输出的 logits 和真实标签 labels，不需要提前手动 softmax。

7. `labels` 为什么通常是 `[batch_size]`，而不是 `[batch_size, num_classes]`？
    因为 `CrossEntropyLoss` 需要的是每个样本的类别编号，而不是 one-hot 概率分布。

## 11. 未理解点与疑问

这一部分不需要重复前文内容，只记录当前还不清楚、需要后续通过代码或资料验证的问题。

### 概念疑问

```text
待补充
```

### 公式疑问

```text
待补充
```

### 代码疑问

```text
待补充
```

### 后续需要验证的点

```text
待补充
```
