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

### MRI-Level Data Organization

Each MRI scan is stored as a folder containing multiple 2D PNG slices (typically over 100 slices per scan). The dataset is organized into two classes:

- KL0 (non-OA)
- KL3/4 (OA)

---

### Train/Test Split (MRI-Level)

The dataset is split into training and testing sets using an **80% / 20% ratio**.

Importantly, the split is performed at the **MRI-folder level**, not at the slice level.

This ensures that all slices from the same MRI scan remain in the same split (either training or testing), preventing data leakage.

Stratified sampling is applied to maintain class balance between KL0 and KL3/4.

---

### Slice Selection (Middle 20 Slices)

Each MRI contains many slices, but not all slices are equally informative. Peripheral slices often contain limited anatomical information.

To focus on the most relevant region, we select only the **middle 20 slices** from each MRI.

Example:

- If an MRI has 160 slices (indexed from 0 to 159)  
- Selected slices: **70 to 89**

This reduces noise and improves training efficiency.

---

### Slice-Level Dataset Construction

After splitting MRI folders into training and testing sets, the selected slices are flattened into individual image samples.

Each slice inherits the label of its corresponding MRI:

- KL0 → label 0  
- KL3/4 → label 1  

Thus, the model is trained and evaluated at the **slice level**, while the split is performed at the **MRI level**.

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

Experiment A: We conduct **6 experiments** to analyze the effect of different training settings.

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
Experiment B: We conduct **3 experiments** to analyze the effect of different slice amounts using the same parameter settings as the Baseline in Experiment A.

### Run 1 — Baseline (20 Slices)

epochs = 20
learning rate = 1e-3
batch size = 8
model = SimpleCNN

### Run 2 — Fewer Slices (10 Slices)

epochs = 20
learning rate = 1e-3
batch size = 8
model = SimpleCNN

Purpose:
- Evaluate the effect of fewer slices on a small dataset

### Run 3 — More Slices (30 Slices)

epochs = 20
learning rate = 1e-3
batch size = 8
model = SimpleCNN

Purpose:
- Evaluate the effect of more slices on a small dataset
---

## Evaluation Metrics

We evaluate model performance using `sklearn`:

- Accuracy  
- Precision  
- Recall (Sensitivity)  
- Specificity  
- F1-score  
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

Using the middle 20 slices from each MRI, we report the main metrics for all 6 experiments in experiment A:

| Experiment | Acc | Recall | Spec | F1 |
|------------|-----|--------|------|-----|
| Baseline | 0.5816 | 0.7278 | 0.4500 | 0.6223 |
| Fewer Epochs | 0.5921 | 0.8000 | 0.4050 | 0.6501 |
| More Epochs | 0.6316 | 0.7667 | 0.5100 | 0.6635 |
| Lower LR | 0.5816 | 0.7389 | 0.4400 | 0.6259 |
| Higher LR | 0.4737 | 1.0000 | 0.0000 | 0.6429 |
| Larger Model | 0.6289 | 0.7167 | 0.5500 | 0.6466 |

Overall, the best accuracy is achieved with more training epochs (0.6316), while the larger model improves specificity (0.55). The model consistently shows higher recall than specificity, indicating better detection of OA cases but more false positives. A high learning rate leads to unstable training and poor performance.




Using the middle 20 slices from each MRI as a baseline, we report the main metrics for the 2 additional experiments (10 and 30 slices) in experiment B:

| Experiment | Acc | Recall | Spec | F1 |
|------------|-----|--------|------|-----|
| Baseline (20) | 0.5711 | 0.7611 | 0.4000 | 0.6270 |
| Fewer Slices (10) | 0.5263 | 0.6667 | 0.4000 | 0.5714 |
| More Slices (30) | 0.5544 | 0.7148 | 0.4100 | 0.6031 |

Confusion Matrix: 20 slices
|               | Pred Normal | Pred OA |
|--------------|--------|------------|
| True Normal      | 80     | 120         |
| True OA  | 43     | 137         |

Confusion Matrix: 10 Slices
|               | Pred Normal | Pred OA |
|--------------|--------|------------|
| True Normal      | 40     | 60         |
| True OA  | 30     | 60         |

Confusion Matrix: 30 Slices
|               | Pred Normal | Pred OA |
|--------------|--------|------------|
| True Normal      | 123     | 177         |
| True OA  | 77     | 193         |

20 slices performed the best and had the highest accuracy, recall, F1, and balanced accuracy. This suggests 20 slices is the most clinically useful amount for this application.

Using the middle 10 slices saw a drop in accuracy, recall, and F1, but had the highest AUC. This implies that too few slices lead to information loss, and the model becomes less stable

Running the model with the 30 middle slices performed slightly better than 10 slices, but did not outperform using the middle 20. This indicates that using more slices introduced irrelevant anatomy, noise, and redundant information.

These results show there appears to be an optimal slice window (20 slices) where the model balances signal versus noise. Too few slices lose critical features, while too many dilute relevant information.

## Limitations

Several limitations should be considered when interpreting these results.

### 1) Small dataset size
The dataset includes only 122 MRI scans, which limits generalizability and increases the risk of the model overfitting.

### 2) Class imbalance
The model consistently shows higher recall than specificity, alluding to a bias toward predicting osteoarthritis (KL 3/4) and producing false positives.

### 3) Single train/test split evaluated
These results are based on a single split (80/20%). This may not fully reflect variablity in model performance.

### 4) Single model exploration
The only model used for this experiment was CNN. Other and/or more advanced models may improve performance and results.

---

## Potential Clinical Applications & Risks

Despite limitations, this experiment demonstrates several potential clinical applications.

### 1) Decision support tools
Providing a second opinion to improve diagnosis consistency.

### 2) Workflow prioritization
Utilizing a predictive model could help prioritize higher risk cases in a clinical setting to improve efficiency and patient outcomes.

### 3) Automated knee osteoarthritis screening platform
A model could aid radiologists by flagging MRIs that require a more in-depth review.

### 4) Research applications
Analysis of knee osteoarthritis imaging datasets can expand research in the field and improve diagnosis and treatments.

The model's low specificity and high false positive rate may limit its direct clinical use and relevance, which could lead to incorrect diagnoses and unncessary evaluations. The model would also need validated on other datasets before being implemented in a clinical setting. At present, this approach is best suited as a decision support tool and not an automated diagnostic system.

---

## Future Work

### 1) Prediction at MRI level instead of slice level
This approach treats each MRI slice independently. Future work could investigate MRI level prediciton or use models that are able to process 3D images.

### 2) Larger and more diverse datasets
Expanding the dataset with more MRI scans from multiple sources would improve generalizability and model performance. Including a wider range of patient demographics and imaging variations would also make the model more robust and more in line with real clinical data.

### 3) Advanced models
This project uses a simple CNN, future work could explore more advanced deep learning architectures and models designed for medical imaging tasks. These models may improve feature extraction and classification performance.

### 4) Different slice selection strategies
Instead of selecting the fixed amount of middle MRI slices, future approaches could use adaptive slice selection methods to identify the most informative and optimal regions of the MRI. This approach could help focus the model on clinically relevant features.

---

## Summary

In this project, we developed a convolutional neural network to classify knee MRI scans into non-osteoarthritis (KL 0) and osteoarthritis (KL 3/4). We explored the effects of training parameters and slice selection amounts on model performance.

Our results showed that increasing epochs and model complexity improved performance. Slice selection experiments indicated that using the middle 20 slices per MRI produced the best overall results. This suggests an optimal balance between capturing important knee anatomy information and reducing noise.

Across all experiments, the model produced higher recall than specificity representing a stronger ability to detect osteoarthritis cases but a tendency toward false positives.

Overall, this project highlights major challenges in medical imaging classification tasks using deep learning models, such as limited datasets, image noise, and the importance of preprocessing choices. Despite our model's modest performance, the findings provide insight into how model design and data selection impact outcomes. This provides a foundation to build upon for future improvements.


