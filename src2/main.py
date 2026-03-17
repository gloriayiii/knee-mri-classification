import os
import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset import KneeMRIDataset
from model import SimpleCNN
from train import train_model
from evaluate import evaluate
import random
import numpy as np
import torch

def set_seed(seed=17):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed set to {seed}")

class DeeperCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


EXPERIMENTS = {
    "baseline": {
        "model_name": "SimpleCNN",
        "epochs": 20,
        "lr": 1e-3,
        "batch_size": 8
    },
    "exp_a_fewer_epochs": {
        "model_name": "SimpleCNN",
        "epochs": 10,
        "lr": 1e-3,
        "batch_size": 8
    },
    "exp_b_more_epochs": {
        "model_name": "SimpleCNN",
        "epochs": 30,
        "lr": 1e-3,
        "batch_size": 8
    },
    "exp_c_lower_lr": {
        "model_name": "SimpleCNN",
        "epochs": 20,
        "lr": 1e-4,
        "batch_size": 8
    },
    "exp_d_higher_lr": {
        "model_name": "SimpleCNN",
        "epochs": 20,
        "lr": 1e-2,
        "batch_size": 8
    },
    "exp_e_larger_model": {
        "model_name": "DeeperCNN",
        "epochs": 20,
        "lr": 1e-3,
        "batch_size": 8
    }
}


def load_data(root_dir):
    kl0_dir = os.path.join(root_dir, "KL0")
    kl34_dir = os.path.join(root_dir, "KL34")

    kl0_paths = sorted([
        os.path.join(kl0_dir, f)
        for f in os.listdir(kl0_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    kl34_paths = sorted([
        os.path.join(kl34_dir, f)
        for f in os.listdir(kl34_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    image_paths = kl0_paths + kl34_paths
    labels = [0] * len(kl0_paths) + [1] * len(kl34_paths)

    print(f"Loaded {len(kl0_paths)} KL0 images and {len(kl34_paths)} KL3/4 images.")
    print(f"Total images: {len(image_paths)}")

    return image_paths, labels


def build_model(model_name):
    if model_name == "SimpleCNN":
        return SimpleCNN()
    elif model_name == "DeeperCNN":
        return DeeperCNN()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def run_experiment(exp_name, config, train_paths, test_paths, train_labels, test_labels):
    print("\n" + "=" * 70)
    print(f"Running experiment: {exp_name}")
    print(config)
    print("=" * 70)

    train_dataset = KneeMRIDataset(train_paths, train_labels)
    test_dataset = KneeMRIDataset(test_paths, test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )

    model = build_model(config["model_name"])

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        epochs=config["epochs"],
        lr=config["lr"]
    )

    metrics = evaluate(model, test_loader, print_result=True)

    result = copy.deepcopy(config)
    result["experiment"] = exp_name
    result["accuracy"] = metrics["accuracy"]
    result["balanced_accuracy"] = metrics["balanced_accuracy"]
    result["precision"] = metrics["precision"]
    result["recall"] = metrics["recall"]
    result["specificity"] = metrics["specificity"]
    result["f1"] = metrics["f1"]
    result["auc"] = metrics["auc"]

    return result


def print_summary_table(results):
    print("\n" + "=" * 110)
    print("Experiment Summary")
    print("=" * 110)
    header = (
        f"{'Experiment':25s}"
        f"{'Model':15s}"
        f"{'Epochs':8s}"
        f"{'LR':10s}"
        f"{'Batch':8s}"
        f"{'Acc':10s}"
        f"{'Bal Acc':10s}"
        f"{'Recall':10s}"
        f"{'Spec':10s}"
        f"{'F1':10s}"
    )
    print(header)
    print("-" * 110)

    for r in results:
        print(
            f"{r['experiment']:25s}"
            f"{r['model_name']:15s}"
            f"{str(r['epochs']):8s}"
            f"{str(r['lr']):10s}"
            f"{str(r['batch_size']):8s}"
            f"{r['accuracy']:.4f}    "
            f"{r['balanced_accuracy']:.4f}    "
            f"{r['recall']:.4f}    "
            f"{r['specificity']:.4f}    "
            f"{r['f1']:.4f}"
        )


def main():
    set_seed(17)
    root_dir = "kl_dataset"

    image_paths, labels = load_data(root_dir)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    print(f"Train size: {len(train_paths)}")
    print(f"Test size: {len(test_paths)}")

    selected_experiments = list(EXPERIMENTS.keys())

    results = []

    for exp_name in selected_experiments:
        config = EXPERIMENTS[exp_name]
        result = run_experiment(
            exp_name,
            config,
            train_paths,
            test_paths,
            train_labels,
            test_labels
        )
        results.append(result)

    print_summary_table(results)


if __name__ == "__main__":
    main()
