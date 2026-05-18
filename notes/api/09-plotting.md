# 09 - 画图记录

## `plt.plot`

作用：绘制曲线。

常用写法：

```python
plt.plot(train_loss_history, label="train loss")
```

```python
epoch_range = range(1, epochs + 1)
plt.plot(epoch_range, test_accuracy_history, label="test accuracy")
```

常见场景：loss 曲线、accuracy 曲线。

注意：如果同时传入 x 和 y，它们长度必须一致。

## `plt.xlabel` / `plt.ylabel` / `plt.title`

作用：设置横轴、纵轴和标题。

常用写法：

```python
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
```

```python
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Test Accuracy")
```

常见场景：让实验图更容易读。

## `plt.legend`

作用：显示图例。

常用写法：

```python
plt.plot(train_loss_history, label="train loss")
plt.plot(test_loss_history, label="test loss")
plt.legend()
```

```python
plt.plot(test_accuracy_history, label="test accuracy")
plt.legend()
```

注意：曲线要设置 `label`，`legend()` 才有内容显示。

## `plt.grid`

作用：显示网格线。

常用写法：

```python
plt.grid(True)
```

```python
plt.grid(False)
```

常见场景：观察曲线数值变化。

## `plt.figure`

作用：新建一张图。

常用写法：

```python
plt.figure()
```

```python
plt.plot(train_loss_history)
plt.figure()
plt.plot(test_accuracy_history)
```

常见场景：分别绘制 loss 和 accuracy。

## `plt.savefig`

作用：保存当前图像。

常用写法：

```python
plt.savefig("loss_curve.png")
```

```python
plt.savefig("results/accuracy_curve.png", dpi=150, bbox_inches="tight")
```

常见场景：把实验结果放入 README。

## 曲线横轴从 1 开始

作用：避免图上出现第 0 轮。

常用写法：

```python
epoch_range = range(1, epochs + 1)
plt.plot(epoch_range, train_loss_history)
```

```python
plt.plot(range(1, len(test_accuracy_history) + 1), test_accuracy_history)
```

常见场景：训练曲线展示。
