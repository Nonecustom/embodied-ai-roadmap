# 04 - 模型层

## `nn.Module`

作用：所有 PyTorch 模型的基础类。

常用写法：

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
```

```python
    def forward(self, x):
        return x
```

常见场景：自定义神经网络。

注意：继承 `nn.Module` 后，通常要写 `super().__init__()`。

## `forward`

作用：定义数据在模型中的流动路径。

常用写法：

```python
def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x
```

```python
logits = model(images)
```

常见场景：模型前向传播。

注意：调用 `model(images)` 时，PyTorch 会自动调用 `forward(images)`。

## `nn.Sequential`

作用：按顺序组合多个网络层。

常用写法：

```python
self.features = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
)
```

```python
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```

常见场景：简单顺序模型。

## `nn.Linear`

作用：全连接层，把输入特征映射成输出特征。

常用写法：

```python
layer = nn.Linear(4, 3)
```

```python
layer = nn.Linear(64 * 8 * 8, 128)
```

常见场景：分类器、MLP、最后输出类别分数。

## `nn.Conv2d`

作用：二维卷积层，用于图片特征提取。

常用写法：

```python
conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
```

```python
conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)
```

常见参数：

| 参数 | 作用 |
|---|---|
| `in_channels` | 输入通道数 |
| `out_channels` | 输出通道数 |
| `kernel_size` | 卷积核大小 |
| `padding` | 边缘补 0 圈数 |
| `stride` | 卷积核滑动步长 |

注意：CIFAR10 彩色图输入通道是 3；`kernel_size=3, padding=1` 可保持宽高不变。

## `nn.ReLU`

作用：激活函数，把负数变成 0，正数保持不变。

常用写法：

```python
nn.ReLU()
```

```python
x = nn.ReLU()(x)
```

常见场景：卷积层或全连接层之后。

## `nn.MaxPool2d`

作用：最大池化，缩小特征图宽高。

常用写法：

```python
nn.MaxPool2d(2)
```

```python
pool = nn.MaxPool2d(kernel_size=2)
```

常见场景：CNN 中降低空间尺寸。

注意：`MaxPool2d(2)` 通常让宽高减半，通道数不变。

## `nn.Flatten`

作用：把多维特征图展平成一维特征向量。

常用写法：

```python
nn.Flatten()
```

```python
x = torch.randn(64, 64, 8, 8)
y = nn.Flatten()(x)
```

常见场景：卷积层连接全连接层前。

注意：默认保留 batch 维度。
