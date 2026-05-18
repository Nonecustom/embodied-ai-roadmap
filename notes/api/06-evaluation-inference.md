# 06 - 评估与推理

## `model.eval()`

作用：切换到评估模式。

常用写法：

```python
model.eval()
```

```python
model.eval()
with torch.no_grad():
    logits = model(images)
```

常见场景：测试集评估、特征提取、推理。

## `torch.no_grad()`

作用：关闭梯度计算，节省显存和计算。

常用写法：

```python
with torch.no_grad():
    logits = model(images)
```

```python
with torch.no_grad():
    feature = model(image_tensor)
```

常见场景：不需要训练参数时。

## `argmax(dim=1)`

作用：从类别分数中找到预测类别。

常用写法：

```python
predictions = logits.argmax(dim=1)
```

```python
correct = (predictions == labels).sum().item()
```

常见场景：分类准确率计算。

注意：`logits` 形状为 `[batch_size, num_classes]` 时，通常用 `dim=1`。

## accuracy 模板

作用：计算分类准确率。

常用写法：

```python
predictions = logits.argmax(dim=1)
correct += (predictions == labels).sum().item()
total += labels.size(0)
accuracy = correct / total
```

```python
print(f"test_acc: {accuracy * 100:.2f}%")
```

常见场景：测试集评估。

## `to(device)`

作用：把模型或张量移动到 CPU / GPU。

常用写法：

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

```python
images = images.to(device)
labels = labels.to(device)
```

常见场景：GPU 加速训练和推理。

注意：模型和数据必须在同一个设备上。

## 推理函数模板

作用：输入数据，输出预测结果。

常用写法：

```python
model.eval()
with torch.no_grad():
    logits = model(images)
    predictions = logits.argmax(dim=1)
```

```python
model.eval()
with torch.no_grad():
    feature = model(image_tensor)
```

常见场景：测试模型、特征提取、后续真实机械臂推理。
