# Revisiting DeepFool: Generalization and Improvement

This repository contains a Python implementation of the SuperDeepFool idea for finding minimal $L_2$ adversarial perturbations. The summary below matches the PDF text you shared and focuses on the core intuition behind the paper.

## What the paper is about

Neural networks can be fooled by tiny, almost invisible changes to an image. These modified inputs are called adversarial examples. The main goal of the paper is to find the smallest possible change that fools the model.

The paper studies robustness under minimal $L_2$ perturbations. In simple terms:

- original image: the clean input
- attacked image: the image after adding a tiny perturbation
- perturbation norm: the size of the change, measured with the $L_2$ norm

Smaller perturbations mean the model is more fragile.

## DeepFool intuition

DeepFool treats the classifier’s decision boundary as locally linear and asks for the shortest path to cross that boundary.

For a binary classifier, the minimum perturbation can be written as a step along the gradient direction, scaled by the model’s confidence and gradient norm.

For the multi-class case:

- $\hat{k} = \arg\max_k f_k(x)$ is the current predicted class
- $w_i = \nabla f_i(x) - \nabla f_{\hat{k}}(x)$
- $f_i = f_i(x) - f_{\hat{k}}(x)$

The closest boundary is selected by minimizing the ratio between the score gap and the gradient difference.

The perturbation step is the basic DeepFool-style update.

## What SuperDeepFool adds

SuperDeepFool keeps the DeepFool step, then adds an extra geometric refinement step.

The idea is:

1. move into the adversarial region,
2. project back carefully,
3. refine the perturbation so it stays small.

This improves the trade-off between speed and minimality.

The paper describes variants such as:

- SDF(1,1): one refinement step, faster
- SDF(1,3): three refinement projections, more accurate

The second setting is presented as a better balance between speed and accuracy.

## Why this matters

The paper compares SuperDeepFool against older methods such as DeepFool, DDN, FAB, FMN, and C&W. The main message is that SuperDeepFool aims to be:

- fast enough to run on larger models,
- accurate enough to find near-minimal perturbations,
- practical for robustness evaluation and adversarial training.

## How this repository is organized

```text
superdeepfool/
  utils.py                 helper functions such as accuracy and norms
  attacks/
    attack.py              base attack class
    DeepFool.py            DeepFool implementation
    SuperDeepFool.py       main SuperDeepFool attack
    DDN.py                 additional L2 attack implementation
    Search.py              post-processing search step
curvature/
  curvature.py             curvature estimation utilities
main.py                    evaluation script on CIFAR-10
results/                   saved logs and outputs
```

## How to run

The main evaluation script loads CIFAR-10, loads a pretrained RobustBench model, runs SuperDeepFool, and writes statistics to `results/cifar10/superdeepfool.log`.

```bash
python main.py
```

For a quicker run:

```bash
python main.py --n-examples 10
```

## What the results mean

The log file reports:

- `Accuracy of original model`: clean accuracy before attack
- `mean_r_l2`: average perturbation size across attacked samples
- `median_r_l2`: median perturbation size
- `length of perturb`: number of attacked examples
- `Time taken`: runtime for the run

Interpretation:

- lower `mean_r_l2` and `median_r_l2` are better for the attacker,
- higher clean accuracy means the original model is stronger on the sampled data,
- shorter runtime means the attack is more practical.

## Notes from the paper summary

- The paper emphasizes minimal perturbation under the $L_2$ norm.
- DeepFool is fast but less accurate.
- FAB, FMN, DDN, and optimization-based methods can be more accurate but slower.
- SuperDeepFool tries to combine speed and accuracy.
- The curvature discussion in the paper suggests a possible extension: curvature-aware robustness analysis and attack refinement.

## Results section from the summary

The summary you provided reports experiments on MNIST, CIFAR-10, and ImageNet, using models such as IBP, PreActResNet-18, WRN-28-10, LeNet, and ResNet-50. The main reported trend is that SuperDeepFool achieves strong minimal-perturbation results with fewer gradient computations than some competing methods.

## Implementation note

This codebase is built with PyTorch and uses RobustBench for pretrained models and dataset loading. The attack logic is written in a TorchAttacks-style API, with a shared base attack class and separate implementations for different attacks.
