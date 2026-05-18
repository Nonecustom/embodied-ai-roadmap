# PyTorch API Notes

> 这是本项目的 API 速查入口。  
> 不按函数名堆积，而是按功能模块查找。

## 高频速查

| 需求 | 常用写法 | 位置 |
|---|---|---|
| 创建张量 | `torch.tensor(...)` | [张量基础](./api/01-tensor-basic.md) |
| 查看形状 | `x.shape` | [张量基础](./api/01-tensor-basic.md) |
| 改变形状 | `x.reshape(...)` | [张量基础](./api/01-tensor-basic.md) |
| 增加 batch 维度 | `x.unsqueeze(0)` | [张量基础](./api/01-tensor-basic.md) |
| 求导 | `loss.backward()` | [自动求导](./api/02-autograd.md) |
| 加载数据 | `DataLoader(...)` | [数据加载](./api/03-data-loading.md) |
| 定义模型层 | `nn.Linear` / `nn.Conv2d` | [模型层](./api/04-model-layers.md) |
| 训练核心 | `zero_grad → backward → step` | [训练流程](./api/05-training-loop.md) |
| 测试准确率 | `argmax(dim=1)` | [评估与推理](./api/06-evaluation-inference.md) |
| 图像预处理 | `transforms.Compose(...)` | [视觉预处理](./api/07-vision-transforms.md) |
| 预训练 ResNet | `models.resnet18(...)` | [预训练视觉模型](./api/08-pretrained-vision-models.md) |
| 画曲线 | `plt.plot(...)` | [画图记录](./api/09-plotting.md) |
| 常见报错 | shape / 拼写 / 设备错误 | [常见错误](./api/10-common-errors.md) |

## 模块目录

1. [张量基础](./api/01-tensor-basic.md)  
   张量创建、shape、reshape、sum、cat、matmul、unsqueeze。

2. [自动求导](./api/02-autograd.md)  
   `requires_grad`、`backward`、`grad`、`no_grad`。

3. [数据加载](./api/03-data-loading.md)  
   `Dataset`、`DataLoader`、`CIFAR10`、batch。

4. [模型层](./api/04-model-layers.md)  
   `nn.Module`、`Linear`、`Conv2d`、`ReLU`、`MaxPool2d`、`Flatten`、`Sequential`。

5. [训练流程](./api/05-training-loop.md)  
   loss、optimizer、SGD、Adam、训练循环、保存训练记录。

6. [评估与推理](./api/06-evaluation-inference.md)  
   `model.eval()`、`torch.no_grad()`、`argmax`、accuracy、device。

7. [视觉预处理](./api/07-vision-transforms.md)  
   `ToTensor`、`Normalize`、`Resize`、`Compose`、图片 shape。

8. [预训练视觉模型](./api/08-pretrained-vision-models.md)  
   ResNet18、去掉分类头、特征提取、CLIP 相关接口。

9. [画图记录](./api/09-plotting.md)  
   loss 曲线、accuracy 曲线、保存图片。

10. [常见错误](./api/10-common-errors.md)  
    训练中常见错误、原因和排查方法。

## 使用建议

查 API 时先想“我现在在做哪类任务”：

```text
张量操作 → 01
求导 → 02
数据集 → 03
模型结构 → 04
训练 → 05
测试 → 06
图像预处理 → 07
ResNet / CLIP → 08
画图 → 09
报错 → 10
```

每个模块只记录本项目常用内容，不追求覆盖所有 PyTorch API。
