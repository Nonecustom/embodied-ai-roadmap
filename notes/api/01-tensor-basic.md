# 01 - 张量基础

## `torch.tensor`

作用：创建 PyTorch 张量。

常用写法：

```python
x = torch.tensor([1, 2, 3])
```

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
```

常见场景：手动构造小数据、测试自动求导。

注意：需要求导时通常使用浮点数。

## `torch.zeros` / `torch.ones`

作用：创建全 0 或全 1 张量。

常用写法：

```python
x = torch.zeros(3, 4)
```

```python
x = torch.ones(2, 3)
```

常见场景：初始化占位数据、测试 shape。

## `torch.randn`

作用：创建服从标准正态分布的随机浮点张量。

常用写法：

```python
X = torch.randn(100, 4)
```

```python
images = torch.randn(64, 3, 32, 32)
```

常见场景：模拟输入特征、测试模型能否前向传播。

## `torch.randint`

作用：创建随机整数张量，常用于模拟分类标签。

常用写法：

```python
y = torch.randint(0, 3, (100,))
```

```python
labels = torch.randint(0, 10, (64,))
```

常见场景：分类任务标签。

注意：范围是左闭右开，`torch.randint(0, 3, ...)` 只会生成 `0, 1, 2`。

## `x.shape`

作用：查看张量形状。

常用写法：

```python
print(x.shape)
```

```python
print(images.shape)
```

常见场景：排查维度错误。

## `reshape`

作用：改变张量形状。

常用写法：

```python
x = torch.arange(12)
y = x.reshape(3, 4)
```

```python
y = x.reshape(2, -1)
```

常见场景：调整输入形状。

注意：元素总数必须保持一致，`-1` 表示自动推断。

## `unsqueeze` / `squeeze`

作用：增加或去掉维度。

常用写法：

```python
image = image.unsqueeze(0)
```

```python
x = x.squeeze(0)
```

常见场景：单张图片 `[3, 224, 224]` 增加 batch 维度变成 `[1, 3, 224, 224]`。

## `sum`

作用：求和。

常用写法：

```python
x.sum()
```

```python
x.sum(dim=0)
x.sum(dim=1)
```

常见场景：统计正确数量、构造标量 loss。

## `torch.cat`

作用：沿已有维度拼接张量。

常用写法：

```python
c = torch.cat((a, b), dim=0)
```

```python
c = torch.cat((a, b), dim=1)
```

常见场景：合并 batch、拼接特征。

注意：除了拼接维度，其他维度必须一致。

## `torch.stack`

作用：新增一个维度后堆叠张量。

常用写法：

```python
x = torch.stack([a, b], dim=0)
```

```python
x = torch.stack([a, b], dim=1)
```

常见场景：把多张单独图片堆成一个 batch。

## `matmul` / `@`

作用：矩阵乘法。

常用写法：

```python
c = a @ b
```

```python
c = torch.matmul(a, b)
```

常见场景：理解线性层、相似度计算。

注意：前一个矩阵列数要等于后一个矩阵行数。
