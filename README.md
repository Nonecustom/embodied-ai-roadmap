# Embodied AI Roadmap

This repository records my learning path toward embodied AI and robot manipulation.

## Goal

Build a low-cost robot arm project using simulation, imitation learning, and real robot experiments.

## Setup

- [Environment setup](./ENVIRONMENT_SETUP.md)

## Stage Progress

| Stage | Topic | Output | Status |
|---|---|---|---|
| 01 | PyTorch basics | CIFAR10 image classification | Done |
| 02 | Vision features | ResNet feature extraction and CLIP matching | Done |
| 03 | Robot simulation | Gymnasium / MuJoCo / ManiSkill demo | Done |
| 04 | Behavior cloning | Toy BC / CartPole BC | In progress |
| 05 | ACT policy | Action chunk dataset | In progress |

## Repository Structure

```text
embodied-ai-roadmap/
  stage01-pytorch-cifar/
  stage02-vision-features/
  stage03-robot-sim/
  stage04-behavior-cloning/
    toy_bc/
    cartpole_bc/
  stage05-act-policy/
  notes/
```

## Papers

- ACT
- Diffusion Policy
- Mobile ALOHA
- OpenVLA
- SmolVLA

## Training Skills
- learning rate
- batch size
- epoch
- train loss / test loss
- overfitting
- weight decay
- data augmentation
- dropout
- early stopping
