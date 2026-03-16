from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate(model, test_loader):

    y_true = []
    y_pred = []

    for images, labels in test_loader:

        outputs = model(images)

        preds = (outputs > 0.5).int()

        y_true.extend(labels.numpy())
        y_pred.extend(preds.detach().numpy())

    acc = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)