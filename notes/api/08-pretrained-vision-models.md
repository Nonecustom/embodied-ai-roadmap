# 08 - 预训练视觉模型

## `torchvision.models.resnet18`

作用：加载 ResNet18 模型。

常用写法：

```python
import torchvision.models as models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
```

```python
model = models.resnet18(weights=None)
```

常见场景：图像分类、特征提取。

注意：`weights=DEFAULT` 表示加载预训练权重。

## `model.fc = nn.Identity()`

作用：去掉 ResNet18 最后的分类层，让模型输出特征向量。

常用写法：

```python
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Identity()
```

```python
feature = model(image_tensor)
print(feature.shape)
```

常见场景：Stage 02 图像特征提取。

注意：ResNet18 去掉 `fc` 后通常输出 `[batch_size, 512]`。

## `nn.Identity`

作用：什么都不做，输入什么就输出什么。

常用写法：

```python
model.fc = nn.Identity()
```

```python
identity = nn.Identity()
y = identity(x)
```

常见场景：替换模型中不想使用的层。

## 特征提取模板

作用：把图片变成 feature vector。

常用写法：

```python
model.eval()
with torch.no_grad():
    feature = model(image_tensor)
```

```python
print(feature.shape)
```

常见场景：图片检索、图文匹配前处理、后续机器人视觉输入。

## CLIP 常见概念

作用：把图片和文本编码到同一个向量空间。

常见概念：

| 名称 | 含义 |
|---|---|
| image embedding | 图片向量 |
| text embedding | 文本向量 |
| similarity | 图片和文本的相似度 |
| prompt | 输入给模型的文本描述 |

常见场景：

```text
图片：桌面上的红色方块
文本：a red block / a blue cup / a box
输出：最相似的文本
```

注意：CLIP 更适合做开放词汇匹配，不是传统固定类别分类器。
