# 03 - 数据加载

## `Dataset`

作用：表示数据集，负责保存样本和标签。

常用写法：

```python
train_dataset = torchvision.datasets.CIFAR10(...)
```

```python
image, label = train_dataset[0]
```

常见场景：读取训练集、测试集。

## `torchvision.datasets.CIFAR10`

作用：加载 CIFAR10 图像分类数据集。

常用写法：

```python
train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)
```

```python
test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform,
)
```

常见参数：

| 参数 | 作用 |
|---|---|
| `root` | 数据保存路径 |
| `train` | `True` 训练集，`False` 测试集 |
| `download` | 是否自动下载 |
| `transform` | 图片预处理 |

注意：数据集文件不要上传 GitHub，放进 `.gitignore`。

## `DataLoader`

作用：按 batch 从 Dataset 中取数据。

常用写法：

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
)
```

```python
for images, labels in train_loader:
    print(images.shape)
    print(labels.shape)
    break
```

常见参数：

| 参数 | 作用 |
|---|---|
| `batch_size` | 每批样本数 |
| `shuffle` | 是否打乱顺序 |
| `num_workers` | 加载数据的子进程数 |

注意：Windows 初学阶段建议 `num_workers=0`；训练集通常 `shuffle=True`，测试集通常 `shuffle=False`。

## batch

作用：一次喂给模型的一批数据。

常用形状：

```text
images: [batch_size, 3, 32, 32]
labels: [batch_size]
```

常见场景：训练循环、评估循环。
