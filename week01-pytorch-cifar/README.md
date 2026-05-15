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
| Baseline | lr=0.1, epoch=10 |![baseline figure](./mini_cf_images/Figure_1.png)|
| Exp 1 | lr=0.01, epoch=10 |![Exp 1](./mini_cf_images/Figure_2.png)|
| Exp 2 | lr=0.1, epoch=50 |![Exp 2](./mini_cf_images/Figure_3.png)|
| Exp 3 | lr=0.01, epoch=50 |![Exp 3](./mini_cf_images/Figure_4.png)|

### Reflection
The experiment shows that as the number of iterations increases, the loss value decreases, while decreasing the learning rate increases the loss value. Therefore, we can conclude that: 
1. more iterations result in a smaller loss value, but it will eventually approach a certain value;
2. decreasing the learning rate does not necessarily lead to a decrease in the loss value, thus requiring careful selection of the learning rate.


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
