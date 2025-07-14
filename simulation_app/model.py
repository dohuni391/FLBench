"""Model definition and training/testing logic."""

from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model(dataset_name: str) -> nn.Module:
    """Return a ResNet-18 model tailored for the specified dataset."""
    if dataset_name.lower() in ["mnist", "femnist"]:
        num_classes = 62 if dataset_name.lower() == "femnist" else 10
        model = torchvision.models.resnet18(num_classes=num_classes)
        # ResNet-18 expects 3-channel images, which our transform provides
    elif dataset_name.lower() == "cifar10":
        model = torchvision.models.resnet18(num_classes=10)
        # Tweak for CIFAR-10's smaller 32x32 images
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return model.to(DEVICE)

def train(net: nn.Module, trainloader: DataLoader, epochs: int, img_key: str, label_key: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch[img_key].to(DEVICE), batch[label_key].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

def test(net: nn.Module, testloader: DataLoader, img_key: str, label_key: str) -> Tuple[float, float]:
    """Evaluate the network on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch[img_key].to(DEVICE), batch[label_key].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def get_weights(net: nn.Module):
    """Extract model parameters as numpy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net: nn.Module, parameters):
    """Apply parameters to a model."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)