# 02 - 自动求导

## `requires_grad=True`

作用：让 PyTorch 追踪这个张量参与的计算，后续可以求导。

常用写法：

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
```

```python
w = torch.randn(3, 4, requires_grad=True)
```

常见场景：理解自动求导、手动构造可训练参数。

## `loss.backward()`

作用：从 loss 开始反向传播，计算梯度。

常用写法：

```python
loss.backward()
```

```python
y = 2 * x ** 2 + 3 * x
y.sum().backward()
```

常见场景：训练模型、测试梯度。

注意：常见情况下 `loss` 应该是标量。

## `.grad`

作用：查看张量或参数的梯度。

常用写法：

```python
print(x.grad)
```

```python
for p in model.parameters():
    print(p.grad)
```

常见场景：理解梯度、排查参数是否参与训练。

## `optimizer.zero_grad()`

作用：清空上一轮梯度。

常用写法：

```python
optimizer.zero_grad()
```

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

常见场景：每个 batch 反向传播前。

注意：PyTorch 默认会累积梯度，所以训练循环里通常必须写。

## `torch.no_grad()`

作用：在该代码块中不计算梯度。

常用写法：

```python
with torch.no_grad():
    logits = model(images)
```

```python
model.eval()
with torch.no_grad():
    feature = model(image)
```

常见场景：测试、推理、特征提取。
