# Implementation Plan - Task 1: Computer Vision using CNN Models on CIFAR-10

This plan outlines the design, execution, and reporting of controlled deep learning experiments on the CIFAR-10 dataset using TensorFlow/Keras on CPU.

## Goal Description
Implement an adapted AlexNet-style CNN for CIFAR-10, run 12 controlled experiments (evaluating regularization, data augmentation, optimization, and network depth), and build a final optimized model that achieves a test accuracy improvement of $\ge 3\%$ over the baseline. The final outputs will be structured in a dedicated directory `task1_cnn/` containing source code, plots, experiment tables, confusion matrices, and a comprehensive research paper report.

---

## Proposed Directory Structure
We will create a directory named `task1_cnn/` containing:
- `src/`: Python source files
  - `data.py`: CIFAR-10 data loading (via Hugging Face to avoid connection resets), preprocessing, and split caching.
  - `models.py`: Keras model definitions (baseline, regularized variants, deeper architectures, and final custom CNN).
  - `train.py`: Utilities for training models with customizable configuration parameters.
  - `run_all_experiments.py`: Master script that runs all experiments sequentially, saves results, and generates all deliverables.
- `plots/`: Automatically generated training history curves (accuracy/loss) and architecture diagrams.
- `results/`: CSV/Markdown tables of results, performance metrics, and confusion matrices.
- `requirements.txt`: Python package requirements.
- `README.md`: Instructions for running the project.
- `REPORT.md`: Comprehensive mini research paper.

---

## Experimental Setup

### 1. Data Splitting & Preprocessing
- **Source**: Hugging Face `uoft-cs/cifar10` (already verified and downloaded).
- **Split**: 40,000 train, 10,000 validation, and 10,000 test.
- **Normalization**: Pixel values normalized to $[0, 1]$.
- **Seed**: Fixed seed $42$ applied to Python, NumPy, and TensorFlow.

### 2. Model Architecture

#### Baseline CNN (AlexNet-style adapted for 32x32)
- Conv2D(32, 3x3, activation='relu', padding='same') -> MaxPooling2D(2x2)
- Conv2D(64, 3x3, activation='relu', padding='same') -> MaxPooling2D(2x2)
- Conv2D(128, 3x3, activation='relu', padding='same') -> MaxPooling2D(2x2)
- Flatten() -> Dense(128, activation='relu') -> Dense(10, activation='softmax')
- **Parameters**: ~350k parameters.
- **Speed**: ~27.5s per epoch on 20,000 samples, ~55s on 40,000 samples.

### 3. Controlled Experiments (Part B)
Each experiment will run for a fixed budget of **10 epochs** to isolate variables fairly.

| Exp ID | Category | Description | Variables Changed |
|---|---|---|---|
| `exp0` | Baseline | Adapted AlexNet-style CNN, Adam ($0.001$), no regularization, no augmentation | None |
| `exp1` | Regularization | Baseline + Dropout (0.3 after pooling, 0.5 in dense) | Added Dropout |
| `exp2` | Regularization | Baseline + Batch Normalization after convolutions | Added BN |
| `exp3` | Regularization | Baseline + L2 Regularization (weight decay $1e-4$) | Added L2 penalty |
| `exp4` | Augmentation | Baseline + Light Augmentation (Horizontal Flip) | Light Augmentation |
| `exp5` | Augmentation | Baseline + Moderate Augmentation (Flip + Rotation 0.1 + Shift 0.1) | Moderate Augmentation |
| `exp6` | Augmentation | Baseline + Aggressive Augmentation (Flip + Rotation 0.2 + Shift 0.2 + Zoom 0.2 + Brightness) | Aggressive Augmentation |
| `exp7` | Optimization | Baseline + SGD + Momentum (learning rate $0.01$, momentum $0.9$) | SGD Optimizer |
| `exp8` | Optimization | Baseline + RMSprop (learning rate $0.001$) | RMSprop Optimizer |
| `exp9` | Optimization | Baseline + Adam (learning rate $0.01$) | Learning Rate = 0.01 |
| `exp10`| Optimization | Baseline + Adam (learning rate $0.0001$) | Learning Rate = 0.0001 |
| `exp11`| Architecture | Baseline + Extra Conv Block (Conv2D 256, 3x3) before flattening | Added Depth |

### 4. Final Customized CNN (Part C)
- Combine winning strategies from Part B (e.g., Best Optimizer/LR + Batch Normalization + Dropout + Moderate Augmentation).
- Verify target: test accuracy improves by $\ge 3\%$ over the baseline.

---

## Deliverables & Automated Verification

Our master script `run_all_experiments.py` will generate:
1. **Master Experiment Table**: Metrics for all experiments in CSV and Markdown formats.
2. **Performance Table**: Accuracy, Precision, Recall, and F1-score for the baseline and final customized models.
3. **Confusion Matrices**: Saved as PNGs, with the most confused class pairs (e.g. cat vs dog, automobile vs truck) identified.
4. **Training Plots**: Loss and accuracy curves for all configurations.
5. **Architecture Diagrams**: Structural diagrams of the baseline and final models.
6. **REPORT.md**: A mini research-paper style report containing sections: Abstract, Introduction (Baseline), Methodology (Experiments), Results & Analysis, Trade-off Analysis, and Conclusion.

## Verification Plan

### Automated Verification
1. Run `python run_all_experiments.py` to train all models and generate all plots, tables, and reports.
2. Verify that `task1_cnn/results/master_experiment_table.md` is populated.
3. Verify that `task1_cnn/plots/` contains loss/accuracy curves.
4. Verify that the final model achieves test accuracy $\ge 3\%$ relative improvement over the baseline.
