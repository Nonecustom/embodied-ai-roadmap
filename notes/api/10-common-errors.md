# 10 - 常见错误

## `Dimension out of range`

现象：

```text
Dimension out of range
```

常见原因：张量维度不够，却使用了不存在的 `dim`。

例子：

```python
x = torch.tensor([1, 2, 3])
x.sum(dim=1)
```

排查：

```python
print(x.shape)
```

## `Sizes of tensors must match`

现象：

```text
Sizes of tensors must match
```

常见原因：拼接或逐元素运算时 shape 不匹配。

例子：

```python
a = torch.randn(2, 4)
b = torch.randn(3, 3)
torch.cat((a, b), dim=0)
```

排查：

```python
print(a.shape)
print(b.shape)
```

## `optimizer got an empty parameter list`

现象：

```text
ValueError: optimizer got an empty parameter list
```

常见原因：

- `__init__` 拼写错误；
- 模型层没有写成 `self.xxx`；
- 没有继承 `nn.Module`；
- 忘记 `super().__init__()`。

排查：

```python
print(sum(p.numel() for p in model.parameters()))
```

## `unexpected keyword argument`

现象：

```text
unexpected keyword argument 'xxx'
```

常见原因：参数名拼写错误。

例子：

```python
plt.plot(x, lable="loss")
```

应改为：

```python
plt.plot(x, label="loss")
```

常见拼写坑：

| 错误 | 正确 |
|---|---|
| `lable` | `label` |
| `shuflle` | `shuffle` |
| `__init___` | `__init__` |

## GPU 不可用

现象：

```python
torch.cuda.is_available()
```

输出：

```text
False
```

常见原因：

- 安装了 CPU 版 PyTorch；
- VS Code 使用了错误 Python 环境；
- NVIDIA 驱动异常。

排查：

```bash
nvidia-smi
```

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## CIFAR10 下载失败

现象：

```text
RuntimeError: File not found or corrupted.
```

常见原因：

- 网络中断；
- 代理导致下载文件错误；
- 数据集压缩包损坏；
- 本地已有坏文件。

处理：

```text
删除损坏数据 → 重新下载
或手动下载并解压 → download=False
```

注意：数据目录不要上传 GitHub。

## x 和 y 长度不一致

现象：

```text
x and y must have same first dimension
```

常见原因：画图时横坐标和纵坐标数量不一样。

例子：

```python
epoch_range = range(1, epochs)
plt.plot(epoch_range, train_loss_history)
```

应改为：

```python
epoch_range = range(1, epochs + 1)
plt.plot(epoch_range, train_loss_history)
```
