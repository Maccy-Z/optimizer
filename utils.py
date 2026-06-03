from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from typing import Tuple


def get_data_loaders(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader


def validate(model: torch.nn.Module, val_loader: DataLoader, criterion: torch.nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    tracking = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, tracking)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / len(val_loader.dataset)
    return avg_loss, accuracy

import torch
import matplotlib.pyplot as plt

def plot_svd_spectrum(matrix, log_scale=True, title="Singular Value Spectrum",
                      xlabel="Index", ylabel="Singular Value",
                      figsize=(8, 5), show_grid=True):
    """
    Compute and plot the SVD spectrum of a matrix.

    Args:
        matrix (torch.Tensor): Input matrix of shape (m, n).
        log_scale (bool): If True, use logarithmic scale on the y-axis.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        figsize (tuple): Figure size (width, height) in inches.
        show_grid (bool): Whether to show grid lines.
    """
    # Ensure the input is a 2D tensor
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {matrix.shape}")

    # Move matrix to CPU for plotting (avoid GPU issues with matplotlib)
    matrix_cpu = matrix.cpu()

    # Compute singular values using torch.linalg.svd
    # S is returned as a 1D tensor of singular values in descending order
    U, S, Vh = torch.linalg.svd(matrix_cpu, full_matrices=False)

    # Convert to numpy for plotting
    singular_values = S.numpy()
    indices = range(1, len(singular_values) + 1)

    # Scale singular value so largest is 1
    singular_values /= singular_values.max()

    # Create the plot
    plt.figure(figsize=figsize)
    plt.plot(indices, singular_values, marker='o', linestyle='-', markersize=4)

    if log_scale:
        plt.yscale('log')
        plt.ylabel(f"{ylabel} (log scale)")
    else:
        plt.ylabel(ylabel)

    plt.xlabel(xlabel)
    plt.title(title)
    if show_grid:
        plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Create a random 5x5 matrix
    A = torch.randn(5, 5)
    plot_svd_spectrum(A, log_scale=False, title="SVD spectrum of a random 5x5 matrix")
