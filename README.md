# knee-mri-classification


# Knee MRI OA Classification (KL 0 vs KL 3/4)

## Overview

This project implements a simple convolutional neural network (CNN) to classify knee MRI images from the Osteoarthritis Initiative (OAI) dataset into:

- **Non-OA** (KL = 0)
- **OA** (KL = 3 or 4)

The goal is not to build a perfect model, but to understand:

- the medical imaging pipeline  
- the impact of model settings  
- how to interpret evaluation metrics  
- limitations of small datasets  

---

## Dataset

- Total samples: 122 MRI images  
  - 61 KL = 0  
  - 61 KL = 3/4
    
---

## Model

We use a simple CNN with:

- 3 convolutional layers  
- ReLU activations  
- Max pooling  
- Fully connected classifier  

Output:
- Single neuron with sigmoid (binary classification)

Loss:
- `BCEWithLogitsLoss`

Optimizer:
- Adam

---

## Training Setup

- Image size: 224 × 224  
- Batch size: 8  
- Train/test split: 80/20 (stratified)  
- Device: CPU or GPU  

---

## Experiments

We evaluate how model performance changes with different settings.

### Baseline

- epochs = 20  
- learning rate = 1e-3  
- batch size = 8  

---

### Exp A: Fewer epochs

- epochs = 10  
- learning rate = 1e-3  
- batch size = 8  

Purpose:
- Check underfitting behavior

---

### Exp B: More epochs

- epochs = 30  
- learning rate = 1e-3  
- batch size = 8  

Purpose:
- Check overfitting behavior

---

### Exp C: Lower learning rate

- epochs = 20  
- learning rate = 1e-4  
- batch size = 8  

Purpose:
- Check training stability and convergence speed

---

### Exp D: Larger model (optional)

- deeper CNN (more channels / layers)  
- epochs = 20  
- learning rate = 1e-3  
- batch size = 8  

Purpose:
- Evaluate effect of model complexity on small dataset

---

## Evaluation Metrics

We compute:

- Accuracy  
- Precision  
- Recall (Sensitivity)  
- Specificity  
- F1-score  
- Confusion Matrix  

---

## Results Interpretation

### Accuracy

Measures overall correctness, but may be misleading in medical tasks.

---

### Sensitivity (Recall)

- Measures how many OA cases are correctly detected  
- Low sensitivity means missed disease cases  

---

### Specificity

- Measures how many healthy cases are correctly identified  
- Low specificity means false alarms  

---

### Confusion Matrix

Provides detailed breakdown:

|               | Pred OA | Pred Normal |
|--------------|--------|------------|
| True OA      | TP     | FN         |
| True Normal  | FP     | TN         |

---

## Expected Observations

- Fewer epochs → underfitting (low accuracy)  
- More epochs → possible overfitting  
- Lower learning rate → slower but more stable training  
- Larger model → may not improve performance due to small dataset  

---

## Limitations

- Very small dataset (122 samples)  
- Risk of overfitting  
- MRI input vs radiographic KL label mismatch  
- No patient-level split (possible data leakage if slices overlap)  
- Simple CNN may miss subtle anatomical features  

---

## Potential Clinical Applications

- Preliminary screening tool for OA  
- Assist radiologists in identifying suspicious cases  
- Could support disease monitoring with more data  

However, the current model is **not suitable for clinical deployment** due to limited data and validation.

---

## Future Work

- Use pretrained models (ResNet, ViT)  
- Add data augmentation  
- Apply Grad-CAM for interpretability  
- Use larger and multi-center datasets  
- Incorporate clinical metadata (age, sex, BMI)  

---

## Summary

This project demonstrates:

- how to build a basic medical imaging classification pipeline  
- how model settings affect performance  
- why multiple evaluation metrics are necessary  
- the challenges of applying deep learning to small medical datasets  
