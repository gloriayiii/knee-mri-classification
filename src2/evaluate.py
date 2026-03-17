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

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = None

    if print_result:
        print(f"Accuracy: {acc:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"F1-score: {f1:.4f}")
        if auc is not None:
            print(f"AUC: {auc:.4f}")

        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))

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