# CIFAR-10 Computer Vision CNN Experiments

This project implements a reproducible pipeline to train, experiment with, and evaluate Convolutional Neural Network (CNN) architectures on the CIFAR-10 image classification task using TensorFlow/Keras.

It establishes a clean baseline CNN (AlexNet-style adapted for 32x32 images), runs a series of 12 controlled experiments across regularization, data augmentation, optimization, and architecture depth, and finally combines the winning strategies to train a highly optimized Customized CNN.

## Directory Structure

```
task1_cnn/
├── src/
│   ├── data.py                 # Data loading from Hugging Face, split caching (40k/10k/10k), normalization
│   ├── models.py               # CNN architecture definitions (baseline, experimental, and final model)
│   ├── train.py                # Training loops, evaluation metrics, plot generators (curves, confusion matrix)
│   ├── generate_diagrams.py    # Custom script generating publication-quality block diagrams for models
│   └── run_all_experiments.py  # Master orchestrator script running all experiments & generating final report
├── plots/
│   ├── confusion_matrices/     # Confusion matrix plots (PNG) for each experiment
│   ├── baseline_architecture.png
│   ├── final_architecture.png
│   └── *_learning_curves.png   # Train/val accuracy and loss history curves
├── results/
│   ├── experiment_metrics.json  # Checkpointed experiment results cache
│   ├── master_experiment_table.csv
│   ├── master_experiment_table.md
│   ├── performance_table.md
│   └── tradeoff_analysis.md
├── requirements.txt            # Python package requirements
├── REPORT.md                   # Mini research-paper report summarizing all findings
└── README.md                   # This instruction file
```

## Setup Instructions

### 1. Requirements
Install the required python packages:
```bash
pip install -r requirements.txt
```

### 2. Running the Experiments
To run the entire suite of baseline and controlled experiments, compile all tables, save curves, and generate the research report, execute the master orchestrator script:
```bash
python src/run_all_experiments.py
```

*Note: Since the script checkpoints results in `results/experiment_metrics.json`, if training gets interrupted, you can rerun the command to resume from where it left off.*

## Deliverables Generated
- **Master Experiment Table**: Metrics for all Part B experiments comparing test accuracy, training time, parameters, and train-val gap.
- **Performance Table**: Accuracy, Precision, Recall, and F1-score comparison between the Baseline and Final models.
- **Confusion Matrices**: Detailed plots identifying class-specific confusions (e.g. cat vs dog, automobile vs truck) for the Baseline and Final models.
- **Trade-off Analysis**: Accuracy gained per extra million parameters vs training time cost.
- **Research Paper Report**: Comprehensive document (`REPORT.md`) structured as a mini research paper.
