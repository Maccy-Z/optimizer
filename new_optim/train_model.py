import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from torch.utils.data import DataLoader

from utils import get_data_loaders, validate
from optim_utils import ManualLinear

torch.set_printoptions(precision=5)
torch.manual_seed(0)


class MLP(nn.Module):
    """A 4-layer MLP for CIFAR-10 classification."""

    def __init__(self, input_size=3 * 32 * 32, hidden_size=1280, num_classes=10, dropout=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x, _):
        return self.model(x)


def train_one_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device, tracking: list) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        if torch.rand(1).item() > 0.5:
            images = torch.flip(images, dims=[3])

        optimizer.zero_grad()
        loss = criterion(model(images, tracking), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def sweep_learning_rates(epochs: int = 5) -> float:
    lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_data_loaders(batch_size=64)
    criterion = nn.CrossEntropyLoss()
    
    best_lr = lrs[0]
    best_val_acc = 0.0

    print(f"Starting learning rate sweep on {device}...")
    for lr in lrs:
        print(f"\nTesting LR: {lr}")
        model = MLP().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        
        for epoch in range(epochs):
            tracking = []
            train_one_epoch(model, train_loader, criterion, optimizer, device, tracking)
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"LR: {lr} | Final Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_lr = lr

    print(f"\nBest LR found: {best_lr} with validation accuracy: {best_val_acc:.4f}")
    return best_lr


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting training on {device}...")

    train_loader, val_loader = get_data_loaders(batch_size=512)
    model = MLP().to(device)
    model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    tracking = []

    epochs = 10
    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, tracking)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.2f}s")

    print("Training finished!")

    torch.save(tracking, "tracking.pt")


if __name__ == '__main__':
    # sweep_learning_rates()
    main()
