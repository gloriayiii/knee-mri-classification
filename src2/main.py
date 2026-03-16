import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import KneeMRIDataset
from model import SimpleCNN
from train import train_model
from eval import evaluate

def load_data(root_dir):
    kl0_dir = os.path.join(root_dir, "KL0")
    kl34_dir = os.path.join(root_dir, "KL34")

    kl0_paths = [os.path.join(kl0_dir, f) for f in os.listdir(kl0_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    kl34_paths = [os.path.join(kl34_dir, f) for f in os.listdir(kl34_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

    image_paths = kl0_paths + kl34_paths
    labels = [0] * len(kl0_paths) + [1] * len(kl34_paths)

    return image_paths, labels

def main():
    root_dir = "kl_dataset"
    image_paths, labels = load_data(root_dir)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    train_dataset = KneeMRIDataset(train_paths, train_labels)
    test_dataset = KneeMRIDataset(test_paths, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = SimpleCNN()

    train_model(model, train_loader, epochs=20, lr=1e-3)
    evaluate(model, test_loader)

if __name__ == "__main__":
    main()
