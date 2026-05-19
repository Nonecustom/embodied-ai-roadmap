# Environment Setup

这份文件用于在一台只有 VS Code 的新 Windows 电脑上，快速恢复当前项目学习环境。

## 1. 必装软件

1. 安装 Git  
   官网：https://git-scm.com/

2. 安装 Python  
   建议安装到：

```text
E:\Python
```

安装后检查：

```powershell
E:\Python\python.exe --version
E:\Python\python.exe -m pip --version
```

3. 安装 VS Code  
   官网：https://code.visualstudio.com/

4. 如果使用 NVIDIA GPU，先安装或更新显卡驱动  
   官网：https://www.nvidia.com/Download/index.aspx

## 2. 配置系统环境变量

打开：

```text
系统属性
-> 高级
-> 环境变量
-> 系统变量
-> Path
-> 编辑
```

建议加入：

```text
E:\Python\
E:\Python\Scripts\
C:\Program Files\Git\cmd
```

如果 Git 安装在其他位置，以实际路径为准。

配置后重新打开 PowerShell，检查：

```powershell
python --version
pip --version
git --version
```

如果 `python` 指向的不是 `E:\Python\python.exe`，优先使用完整命令：

```powershell
E:\Python\python.exe
```

## 3. VS Code 扩展

在 VS Code 扩展栏搜索并安装：

```text
Python
Pylance
Jupyter
C/C++
Git Graph
Markdown All in One
```

可选：

```text
Chinese (Simplified) Language Pack
```

## 4. 克隆项目

```powershell
cd "E:\github clone repo"
git clone https://github.com/Nonecustom/embodied-ai-roadmap.git
cd "E:\github clone repo\embodied-ai-roadmap"
```

## 5. 安装 Python 常用库

先升级基础工具：

```powershell
E:\Python\python.exe -m pip install --upgrade pip setuptools wheel --no-cache-dir
```

安装通用库：

```powershell
E:\Python\python.exe -m pip install numpy matplotlib pillow pandas tqdm --no-cache-dir
```

安装 Gymnasium：

```powershell
E:\Python\python.exe -m pip install gymnasium --no-cache-dir
```

安装 CLIP 相关库：

```powershell
E:\Python\python.exe -m pip install transformers accelerate safetensors --no-cache-dir
```

## 6. 安装 PyTorch

不要固定抄旧命令。进入 PyTorch 官网，根据系统、pip、Python、CUDA 版本选择安装命令：

```text
https://pytorch.org/get-started/locally/
```

当前项目使用过 CUDA 版 PyTorch。安装后必须验证：

```powershell
E:\Python\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

期望：

```text
torch.cuda.is_available() -> True
能打印 NVIDIA GPU 名称
```

如果没有 NVIDIA GPU，可以安装 CPU 版 PyTorch。

## 7. VS Code 选择解释器

打开项目文件夹后：

```text
Ctrl + Shift + P
Python: Select Interpreter
选择 E:\Python\python.exe
```

之后在 VS Code 终端检查：

```powershell
python -c "import sys; print(sys.executable)"
```

应输出：

```text
E:\Python\python.exe
```

## 8. 项目验证

验证 Stage 01：

```powershell
cd "E:\github clone repo\embodied-ai-roadmap\stage01-pytorch-cifar"
E:\Python\python.exe train.py
```

验证 Stage 02：

```powershell
cd "E:\github clone repo\embodied-ai-roadmap\stage02-vision-features"
E:\Python\python.exe resnet_feature.py
E:\Python\python.exe clip_match.py
```

验证 Stage 03：

```powershell
cd "E:\github clone repo\embodied-ai-roadmap\stage03-robot-sim"
E:\Python\python.exe gym_env.py
E:\Python\python.exe random_multi_episode.py
```

## 9. 常见问题

如果 CUDA 不可用：

```text
1. 检查是否安装 NVIDIA 驱动
2. 检查 PyTorch 是否为 CUDA 版
3. 确认 VS Code 使用的是 E:\Python\python.exe
```

如果包安装到了 C 盘：

```text
确认命令使用的是 E:\Python\python.exe -m pip
安装时加 --no-cache-dir 减少 C 盘缓存
```

如果 `python` 命令无法识别：

```text
检查 Path 是否包含 E:\Python\
修改环境变量后要重新打开终端
```

如果 `git` 命令无法识别：

```text
检查 Path 是否包含 C:\Program Files\Git\cmd
或重新安装 Git 并勾选加入 PATH
```

如果 Hugging Face 模型下载失败：

```text
检查网络或代理
稍后重试
不要反复改代码
```

## 10. 整块 E 盘迁移到新电脑

如果把当前 E 盘固态整体装到新电脑，并且新电脑仍识别为 `E:`，通常可以保留：

```text
E:\Python
E:\github clone repo
E:\Deep learning
E:\Python\Lib\site-packages
```

优先不要全部重装，按顺序检查：

```powershell
E:\Python\python.exe --version
E:\Python\python.exe -m pip --version
E:\Python\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
E:\Python\python.exe -c "import gymnasium; print(gymnasium.__version__)"
E:\Python\python.exe -c "import mujoco; print(mujoco.__version__)"
```

新电脑通常仍需要重新配置：

```text
系统 Path：
  E:\Python\
  E:\Python\Scripts\
  C:\Program Files\Git\cmd

VS Code：
  重新安装或确认可用
  重新选择解释器 E:\Python\python.exe

Git：
  重新安装或确认 git --version 可用

NVIDIA：
  重新安装或更新显卡驱动
```

如果新电脑把固态识别成其他盘符，例如 `D:`，建议优先在磁盘管理中改回 `E:`。  
如果不改盘符，很多旧路径需要重新配置，部分 Python 包或 VS Code 设置可能失效。

判断是否需要重装包：

```text
能 import，就先不重装
不能 import，再用 E:\Python\python.exe -m pip install ...
torch.cuda.is_available() 为 False 时，优先检查显卡驱动和 PyTorch CUDA 版本
```

## 11. 最小恢复顺序

```text
Git
Python
配置系统 Path
VS Code
VS Code 扩展
clone 仓库
安装 PyTorch
安装 numpy / matplotlib / pillow / torchvision / transformers / gymnasium
选择 VS Code Python 解释器
逐个运行 Stage 01 / Stage 02 / Stage 03 验证脚本
```
