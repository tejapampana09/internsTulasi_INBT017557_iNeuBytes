# Master Experiment Results Table — CIFAR-10 CNN Study
# Task 1 | Intern: Tulasi | ID: INBT017557

| # | Experiment | Test Acc | Test Loss | Precision | Recall | F1 | Train-Val Gap | Params | Time |
|:--|:-----------|:--------:|:---------:|:---------:|:------:|:--:|:---:|:------:|:----:|
| 0 | Baseline (No Reg, No Aug) | **74.14%** | 0.783 | 73.96% | 74.14% | 73.83% | 8.56% | 356,810 | 615s |
| 1 | Baseline + Dropout | 71.62% | 0.817 | 71.54% | 71.62% | 71.20% | -5.72% | 356,810 | 696s |
| 2 | Baseline + Batch Normalization | 72.78% | 1.027 | 72.84% | 72.78% | 72.54% | 23.64% | 358,218 | 1161s |
| 3 | Baseline + L2 Regularization | 70.51% | 0.913 | 72.05% | 70.51% | 70.33% | 5.89% | 356,810 | 616s |
| 4 | Baseline + Light Augmentation | 71.00% | 0.847 | 71.81% | 71.00% | 70.85% | 5.54% | 356,810 | 697s |
| 5 | Baseline + Moderate Augmentation | 65.75% | 0.977 | 66.78% | 65.75% | 64.34% | -2.76% | 356,810 | 820s |
| 6 | Baseline + Aggressive Augmentation | 56.00% | 1.237 | 57.42% | 56.00% | 54.04% | -5.27% | 356,810 | 902s |
| 7 | SGD + Momentum (lr=0.01) | 65.74% | 0.966 | 68.13% | 65.74% | 65.43% | 3.97% | 356,810 | 569s |
| 8 | RMSprop (lr=0.001) | 70.20% | 0.861 | 71.33% | 70.20% | 70.04% | 5.29% | 356,810 | 624s |
| 9 | Adam (lr=0.01) | 54.53% | 1.276 | 54.91% | 54.53% | 53.95% | 3.05% | 356,810 | 663s |
| 10 | Adam (lr=0.0001) | 57.60% | 1.202 | 56.86% | 57.60% | 57.04% | 0.18% | 356,810 | 778s |
| 11 | Deeper CNN (Extra Block) | 73.26% | 0.793 | 73.86% | 73.26% | 73.08% | 10.04% | 653,386 | 1336s |
| 12 | Final Customized CNN | 67.13% | 0.939 | 68.04% | 67.13% | 66.52% | -3.96% | 356,810 | 874s |

---

## Summary
- **Best Test Accuracy**: exp0 — Baseline @ **74.14%**
- **Best Generalization**: exp1 — Dropout (Train-Val Gap = -5.72%)
- **Best Optimizer**: Adam lr=0.001 (exp0)
- **Worst Accuracy**: exp9 — Adam lr=0.01 @ 54.53% (LR too high)
- **Least Overfitting**: exp10 — Adam lr=0.0001 (Gap = 0.18%)
- **Most Overfit**: exp2 — Batch Normalization (Gap = 23.64%)