import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import KneeMRIDataset
from model import SimpleCNN
from train import train_model
from evaluate import evaluate


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

# Baseline Run 20 Slices

def select_middle_slices(png_paths, num_slices=20):
    png_paths = sorted(png_paths)
    n = len(png_paths)

    if n == 0:
        return []

    if n <= num_slices:
        return png_paths

    start = (n - num_slices) // 2
    end = start + num_slices

    print(f"Total slices: {n}, selecting indices [{start}:{end}]")
    return png_paths[start:end]


def load_mri_groups(root_dir, num_slices=20):
    kl0_dir = os.path.join(root_dir, "KL0")
    kl34_dir = os.path.join(root_dir, "KL34")

    if not os.path.isdir(kl0_dir):
        raise FileNotFoundError(f"KL0 folder not found: {kl0_dir}")
    if not os.path.isdir(kl34_dir):
        raise FileNotFoundError(f"KL34 folder not found: {kl34_dir}")

    mri_groups = []

    for folder_name in sorted(os.listdir(kl0_dir)):
        folder_path = os.path.join(kl0_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        png_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(".png")
        ]
        png_paths = select_middle_slices(png_paths, num_slices=num_slices)

        if len(png_paths) > 0:
            mri_groups.append({
                "folder": folder_path,
                "label": 0,
                "png_paths": png_paths
            })

    for folder_name in sorted(os.listdir(kl34_dir)):
        folder_path = os.path.join(kl34_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        png_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(".png")
        ]
        png_paths = select_middle_slices(png_paths, num_slices=num_slices)

        if len(png_paths) > 0:
            mri_groups.append({
                "folder": folder_path,
                "label": 1,
                "png_paths": png_paths
            })

    num_kl0 = sum(1 for g in mri_groups if g["label"] == 0)
    num_kl34 = sum(1 for g in mri_groups if g["label"] == 1)

    print(f"Loaded {num_kl0} KL0 MRI folders and {num_kl34} KL3/4 MRI folders.")
    print(f"Total MRI folders: {len(mri_groups)}")
    print(f"Using middle {num_slices} slices per MRI.")

    return mri_groups


def split_by_mri_groups(mri_groups, test_size=0.2, random_state=42):
    group_labels = [g["label"] for g in mri_groups]

    train_groups, test_groups = train_test_split(
        mri_groups,
        test_size=test_size,
        stratify=group_labels,
        random_state=random_state
    )

    return train_groups, test_groups


def flatten_groups_to_images(groups):
    image_paths = []
    labels = []

    for g in groups:
        image_paths.extend(g["png_paths"])
        labels.extend([g["label"]] * len(g["png_paths"]))

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
    print("\n" + "=" * 115)
    print("Experiment Summary")
    print("=" * 115)
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
    print("-" * 115)

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

    root_dir = r"C:\Documents\AI Med Imaging\2459 Midterm Project\data\knee_MRIs"
    num_slices = 20

    mri_groups = load_mri_groups(root_dir, num_slices=num_slices)

    train_groups, test_groups = split_by_mri_groups(
        mri_groups,
        test_size=0.2,
        random_state=42
    )

    train_paths, train_labels = flatten_groups_to_images(train_groups)
    test_paths, test_labels = flatten_groups_to_images(test_groups)

    print(f"Train MRI folders: {len(train_groups)}")
    print(f"Test MRI folders: {len(test_groups)}")
    print(f"Train PNG slices: {len(train_paths)}")
    print(f"Test PNG slices: {len(test_paths)}")

    # selected_experiments = ["baseline"]
    selected_experiments = ["baseline"]

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