# week 01
## mini_classification
I built a small classification example with random data.

### Code Structure

- `X`: input features, shape `[100, 4]`
- `y`: labels, shape `[100]`
- `model`: maps 4 input features to 3 class scores
- `loss_fn`: computes classification loss
- `optimizer`: updates model parameters
- `epoch`: repeats the training process

### Experiments

| Experiment | Change | Observation |
|---|---|---|
| Baseline | lr=0.1, epoch=10 |![baseline figure](embodied-ai-roadmap\week01-pytorch-cifar\Figure_1.png)|
| Exp 1 | lr=0.01, epoch=10 |![Exp 1](embodied-ai-roadmap\week01-pytorch-cifar\Figure_2.png)|
| Exp 2 | lr=0.1, epoch=50 |![Exp 2](embodied-ai-roadmap\week01-pytorch-cifar\Figure_3.png)|
| Exp 3 | lr=0.01, epoch=50 |![Exp 3](embodied-ai-roadmap\week01-pytorch-cifar\Figure_4.png)|
### Reflection

In this example, I learned that a classification model outputs logits, and CrossEntropyLoss compares these logits with class labels.


## Week 01 - PyTorch CIFAR10

### goal

Run a basic image classification training pipeline using PyTorch.

### What I Did

- Loaded CIFAR10 dataset
- Trained a CNN model
- Tested the model accuracy
- Changed one training parameter and compared results

### Key Concepts

- Dataset:
- DataLoader:
- Model:
- Loss:
- Optimizer:
- Training loop:

### Results

| Experiment | Change | Accuracy |
|---|---|---|
| Baseline | Original settings | |
| Exp 1 | Changed one parameter | |

### Reflection

This week I learned:
