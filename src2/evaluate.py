import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score
)

def evaluate(model, test_loader, print_result=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(1)

            preds = (probs > 0.5).int()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # === sklearn metrics ===
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)   # sensitivity
    f1 = f1_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    # specificity = recall of negative class
    specificity = recall_score(y_true, y_pred, pos_label=0)

    # AUC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = None

    if print_result:
        print("Accuracy:", acc)
        print("Balanced Accuracy:", bal_acc)
        print("Precision:", precision)
        print("Recall (Sensitivity):", recall)
        print("Specificity:", specificity)
        print("F1-score:", f1)
        if auc is not None:
            print("AUC:", auc)

        print("\nConfusion Matrix:\n", cm)

        print("\nClassification Report:\n",
              classification_report(y_true, y_pred, digits=4))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm
    }
