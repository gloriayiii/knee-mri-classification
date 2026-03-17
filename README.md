# Knee MRI Osteoarthritis Classification (KL 0 vs KL 3/4)

## Overview

This project implements a basic deep learning pipeline to classify knee MRI images into:

- **Non-Osteoarthritis (KL = 0)**
- **Osteoarthritis (KL = 3 or 4)**

using a convolutional neural network (CNN).

The goal of this project is not to build a highly optimized model, but to:

- understand the medical imaging workflow  
- explore how model settings affect performance  
- interpret evaluation metrics (accuracy, confusion matrix, etc.)  
- analyze limitations of small medical datasets  

---

## Dataset

The dataset contains knee MRI scans from the Osteoarthritis Initiative (OAI).

- Total MRI scans: 122  
  - 61 KL = 0  
  - 61 KL = 3 or 4  

Each MRI scan is stored as a **folder of 2D PNG slices** (typically 100+ slices per scan).


---

## Data Processing

### MRI-level Split (Important)

To avoid **data leakage**, the dataset is split at the **MRI folder level**, not at the slice level.

This ensures that slices from the same MRI do not appear in both training and testing sets.

---

### Slice Selection (Middle 20 Slices)

Each MRI contains many slices, but not all slices are equally informative.

We select only the **middle 20 slices** from each MRI:

\[
\text{start} = \frac{N - 20}{2}, \quad \text{end} = \text{start} + 20
\]

Example:

- If an MRI has 160 slices (indexed 0–159)  
- Selected slices: **70–89**

This reduces noise and focuses on the most relevant anatomical region.

---

## Model

We use a simple CNN architecture:

- 3 convolutional layers  
- ReLU activation  
- Max pooling  
- Fully connected layers  
- Output: single neuron (binary classification)

Loss function:
- `BCEWithLogitsLoss`

Optimizer:
- Adam

---

## Experiments

We conduct **6 experiments** to analyze the effect of different training settings.

### Run 1 — Baseline

- epochs = 20  
- learning rate = 1e-3  
- batch size = 8  
- model = SimpleCNN  

---

### Run 2 — Fewer Epochs

- epochs = 10  
- learning rate = 1e-3  
- batch size = 8  

Purpose:
- Examine **underfitting**

---

### Run 3 — More Epochs

- epochs = 30  
- learning rate = 1e-3  
- batch size = 8  

Purpose:
- Examine potential **overfitting**

---

### Run 4 — Lower Learning Rate

- epochs = 20  
- learning rate = 1e-4  
- batch size = 8  

Purpose:
- Evaluate slower but more stable convergence

---

### Run 5 — Higher Learning Rate

- epochs = 20  
- learning rate = 1e-2  
- batch size = 8  

Purpose:
- Evaluate training instability

---

### Run 6 — Larger Model

- deeper CNN (more layers and channels)  
- epochs = 20  
- learning rate = 1e-3  
- batch size = 8  

Purpose:
- Evaluate the effect of model complexity on a small dataset

---

## Evaluation Metrics

We evaluate model performance using `sklearn`:

- Accuracy  
- Balanced Accuracy  
- Precision  
- Recall (Sensitivity)  
- Specificity  
- F1-score  
- AUC (ROC)  
- Confusion Matrix  

### Confusion Matrix

|               | Pred OA | Pred Normal |
|--------------|--------|------------|
| True OA      | TP     | FN         |
| True Normal  | FP     | TN         |

---

## Reproducibility

All experiments use a fixed random seed 17

This ensures consistent train/test splits and model initialization.

---

## Results Analysis

??

## Limitations

??

---

## Potential Clinical Applications

??

---

## Future Work

??

---

## Summary

??


