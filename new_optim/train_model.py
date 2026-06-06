import torch
import torch.nn as nn
import math
import time
from torch.utils.data import DataLoader

from utils import get_data_loaders, validate
from optim_utils import ManualLinear, MuonWithAuxAdam, get_muon_adam_params, StateHolder

torch.set_printoptions(precision=5)
torch.manual_seed(0)


class MLP(nn.Module):
    """A 4-layer MLP for CIFAR-10 classification, using ManualLinear instead of nn.Linear."""

    def __init__(self, input_size=3 * 32 * 32, hidden_size=1280, num_classes=10, dropout=0.2):
        super().__init__()
        self.flatten = nn.Flatten()

        # Layer 1: input -> hidden (with bias)
        self.w1 = nn.Parameter(torch.empty(hidden_size, input_size))
        self.b1 = nn.Parameter(torch.zeros(hidden_size))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        # Layer 2: hidden -> hidden (no bias)
        self.w2 = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.drop2 = nn.Dropout(dropout)

        # Layer 3: hidden -> hidden (no bias)
        self.w3 = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.kaiming_uniform_(self.w3, a=math.sqrt(5))

        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.drop3 = nn.Dropout(dropout)

        # Layer 4: hidden -> num_classes (with bias)
        self.w4 = nn.Parameter(torch.empty(num_classes, hidden_size))
        self.b4 = nn.Parameter(torch.zeros(num_classes))
        nn.init.kaiming_uniform_(self.w4, a=math.sqrt(5))

    def forward(self, x, tracking=None):
        x = self.flatten(x)
        x = ManualLinear.apply(x, self.w1, self.b1, None)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = ManualLinear.apply(x, self.w2, None, tracking)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = ManualLinear.apply(x, self.w3, None, tracking)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.drop3(x)
        x = ManualLinear.apply(x, self.w4, self.b4, None)
        return x


def train_one_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device, tracking: StateHolder) -> float:
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


def main() -> None:
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting training on {device}...")

    train_loader, val_loader = get_data_loaders(batch_size=512)
    model = MLP().to(device)
    # model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    muon_params, adam_params = get_muon_adam_params(model)
    tracking = StateHolder(muon_params)

    muon_dict = dict(params=muon_params, use_muon=True, lr=0.75e-3, momentum=0.9, weight_decay=1e-4)
    adam_dict = dict(params=adam_params, use_muon=False, lr=3e-4, weight_decay=1e-4)
    optimizer = MuonWithAuxAdam([adam_dict, muon_dict], tracking)


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
