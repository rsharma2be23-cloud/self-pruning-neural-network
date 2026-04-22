# Self-Pruning Neural Network (CIFAR-10)

## Overview

This project implements a neural network that learns to prune its own weights during training using learnable gates.
Instead of removing parameters after training, the model dynamically suppresses less important connections while learning.

---

## Approach

* Custom `PrunableLinear` layer with learnable gate parameters
* Gates passed through a sigmoid function to scale weights between 0 and 1
* Sparsity enforced using L1 regularization on gate values

Loss function:

```
Loss = CrossEntropy + λ × Sparsity Loss
```

* CNN backbone + prunable fully connected layers
* Dataset: CIFAR-10
* Framework: PyTorch

---

## Results Summary

| Lambda | Test Accuracy | Sparsity |
| ------ | ------------- | -------- |
| 1e-5   | 81.6%         | 62.5%    |
| 1e-4   | 82.3%         | 87.5%    |
| 1e-3   | 79.8%         | 99.4%    |

---

## Key Observations

* **Low λ (1e-5):**
  Model focuses on accuracy, minimal pruning

* **Medium λ (1e-4):**
  Best balance — high accuracy with strong sparsity

* **High λ (1e-3):**
  Aggressive pruning (~99%), slight drop in accuracy

---

## Training Dynamics

![Training](results/training_1e-05.png)

* Loss decreases smoothly
* Accuracy stabilizes around ~80%
* Sparsity increases after initial learning phase

---

## Gate Value Distribution

![Gates](results/gates_1e-05.png)

* Large spike near **0 → pruned weights**
* Remaining values represent important connections
* Clear separation between useful and redundant parameters

---

## Layer-wise Sparsity (Insight)

* λ = 1e-5 → [62.1%, 69.1%, 59.5%, 29.7%]
* λ = 1e-4 → [87.4%, 90.6%, 84.1%, 36.8%]
* λ = 1e-3 → [99.6%, 97.9%, 96.7%, 68.5%]

Different layers adapt differently, showing that pruning is learned selectively.

---

## Key Insight

Even with extreme pruning:

* Up to **99% sparsity achieved**
* Only ~**1.8% drop in accuracy**

This demonstrates that many parameters in the network are redundant.

---

## How to Run

```bash
pip install torch torchvision matplotlib
```

Open the notebook and run all cells.

---

## Notes

* Pruning happens **during training**, not after
* The model first learns features, then removes weak connections
* Results clearly show the **accuracy vs sparsity trade-off**

---

## Author

Rayan Sharma
