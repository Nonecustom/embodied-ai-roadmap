# 07 - 视觉预处理

## `transforms.Compose`

作用：把多个图片预处理步骤组合起来。

常用写法：

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])
```

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```

常见场景：定义图片进入模型前的处理流程。

## `transforms.ToTensor`

作用：把图片转成 PyTorch Tensor，并把像素值从 `0~255` 转成 `0~1`。

常用写法：

```python
transforms.ToTensor()
```

```python
image_tensor = transforms.ToTensor()(image)
```

常见场景：所有图像模型输入前。

注意：图片形状会从 `[H, W, C]` 变成 `[C, H, W]`。

## `transforms.Normalize`

作用：按通道标准化图片。

公式：

```text
新值 = (原值 - mean) / std
```

常用写法：

```python
transforms.Normalize((0.5, 0.5, 0.5),
                     (0.5, 0.5, 0.5))
```

```python
transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)
```

常见场景：CIFAR10、ImageNet 预训练模型输入。

注意：ResNet 这类 ImageNet 预训练模型通常使用 ImageNet 的 mean/std。

## `transforms.Resize`

作用：调整图片大小。

常用写法：

```python
transforms.Resize((224, 224))
```

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```

常见场景：ResNet、CLIP 等预训练模型通常需要固定输入大小。

## `PIL.Image.open`

作用：读取本地图片。

常用写法：

```python
from PIL import Image
image = Image.open("images/test.jpg")
```

```python
image = Image.open(image_path).convert("RGB")
```

常见场景：加载自己的测试图片。

注意：建议 `.convert("RGB")`，避免灰度图或透明通道导致输入通道不一致。

## 图片 shape 常见变化

```text
PIL 图片
→ ToTensor
[3, H, W]
→ unsqueeze(0)
[1, 3, H, W]
→ 模型
[1, num_classes] 或 [1, feature_dim]
```
