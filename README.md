# Self-Pruning Neural Network (CIFAR-10)

## Overview

This project explores a neural network that can **prune itself during training** instead of relying on post-processing techniques.
The idea is simple: each weight is controlled by a learnable gate, and the model learns which connections are important and which can be removed.

---

## Approach

* Built a custom `PrunableLinear` layer with learnable gate parameters

* Applied a **sigmoid** on gate scores to scale weights between 0 and 1

* Introduced sparsity using **L1 regularization on gate values**

* Total loss:

  `Loss = CrossEntropy + λ × Sparsity Loss`

* Used a CNN backbone followed by prunable fully connected layers

* Dataset: **CIFAR-10**

* Framework: **PyTorch**

---

## Results (λ = 1e-5)

* **Train Accuracy:** 87.17%
* **Validation Accuracy:** 79.66%
* **Sparsity:** 62.49%

Even with a relatively small λ, the model was able to prune a significant portion of weights while maintaining good accuracy.

---

## Training Behavior

![Training](results/training_1e-05.png)

* Loss decreases steadily across epochs
* Validation accuracy stabilizes around ~79–80%
* Sparsity starts increasing later in training and then rises consistently

---

## Gate Distribution

![Gates](results/gates_1e-05.png)

* Large concentration of gate values near **0** → pruned weights
* Remaining gates spread out → important connections retained
* Clear separation between useful and redundant parameters

---

## Observations

* Sparsity does not increase immediately — the model first learns useful features
* After that, regularization pushes weaker connections toward zero
* There is a clear **trade-off between accuracy and sparsity**, even at low λ

---

## How to Run

```bash
pip install torch torchvision matplotlib
```

Then open the notebook and run all cells.

---

## Notes

This implementation focuses on:

* Keeping the model simple and interpretable
* Demonstrating dynamic pruning during training
* Visualizing how sparsity evolves over time

---

## Author

Rayan Sharma
