# 05 - 训练流程

## 训练核心模板

作用：完成一次 batch 的参数更新。

常用写法：

```python
logits = model(images)
loss = loss_fn(logits, labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

```python
for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)
    logits = model(images)
    loss = loss_fn(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

常见场景：所有监督学习训练代码。

## `nn.CrossEntropyLoss`

作用：多分类任务损失函数。

常用写法：

```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
```

```python
logits = torch.randn(32, 10)
labels = torch.randint(0, 10, (32,))
loss = loss_fn(logits, labels)
```

常见场景：图像分类、多类别分类。

注意：输入是 logits，不需要手动 softmax。

## `torch.optim.SGD`

作用：随机梯度下降优化器。

常用写法：

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
)
```

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    weight_decay=1e-4,
)
```

常见参数：

| 参数 | 作用 |
|---|---|
| `lr` | 学习率 |
| `momentum` | 动量 |
| `weight_decay` | 权重衰退，减少过拟合 |

## `torch.optim.Adam`

作用：常用自适应优化器，后续深度学习和机器人学习代码常见。

常用写法：

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5,
)
```

常见场景：Transformer、VLM、模仿学习训练。

## 训练一轮函数模板

作用：遍历训练集所有 batch，返回平均训练 loss。

常用写法：

```python
def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)
```

常见场景：把训练逻辑封装清楚。

## 记录训练历史

作用：保存每轮 loss / accuracy，后续画曲线。

常用写法：

```python
train_loss_history.append(train_loss)
```

```python
test_loss_history.append(test_loss)
test_accuracy_history.append(test_accuracy)
```

常见场景：实验记录、README 结果展示。
