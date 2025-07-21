"""Federated Learning Model and Data Utilities."""

from collections import OrderedDict
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader

# Use CUDA if available, otherwise fall back to CPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------
# Model Definition and Tweaks
# ---------------------------------
def _tweak_resnet_for_cifar10(model: nn.Module) -> nn.Module:
    """Adjust ResNet-18 for CIFAR-10's 32x32 images."""
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def get_model(dataset_name: str, num_classes: int) -> nn.Module:
    """Return a ResNet-18 model tailored for the specified dataset."""
    model = torchvision.models.resnet18(num_classes=num_classes)
    if dataset_name == "cifar10":
        model = _tweak_resnet_for_cifar10(model)
    return model.to(DEVICE)


# ---------------------------------
# Data Loading and Partitioning
# ---------------------------------
DATASET_CONFIG = {
    "mnist": {
        "dataset_id": "ylecun/mnist",
        "img_key": "image",
        "label_key": "label",
        "num_classes": 10,
        "base_transform": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
        "special_partitioning": False,
    },
    "cifar10": {
        "dataset_id": "uoft-cs/cifar10",
        "img_key": "img",
        "label_key": "label",
        "num_classes": 10,
        "base_transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        "special_partitioning": False,
    },
    "femnist": {
        "dataset_id": "flwrlabs/femnist",
        "img_key": "image",
        "label_key": "character",
        "num_classes": 62,
        "base_transform": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
        "special_partitioning": True,
    },
}


def get_dataloaders(
    dataset_name: str, partition_id: int, num_partitions: int, batch_size: int
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Return train/test DataLoaders and config for a given client and dataset."""
    name = dataset_name.lower()
    if name not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    config = DATASET_CONFIG[name]

    def apply_transforms(batch: Dict) -> Dict:
        """Apply transformations to a batch of data."""
        img_transform = config["base_transform"]
        # ResNet18 requires 3-channel images
        batch[config["img_key"]] = [
            img_transform(img.convert("RGB")) for img in batch[config["img_key"]]
        ]
        return batch

    if config["special_partitioning"]:
        fds = FederatedDataset(dataset=config["dataset_id"], partitioners={"train": 1})
        full_dataset = fds.load_split("train")
        train_test_split = full_dataset.train_test_split(test_size=0.2, seed=42)
        train_set = train_test_split["train"]
        test_set = train_test_split["test"]
        partition_size = len(train_set) // num_partitions
        start, end = partition_id * partition_size, (partition_id + 1) * partition_size
        partition = train_set.select(range(start, end))
    else:
        fds = FederatedDataset(
            dataset=config["dataset_id"], partitioners={"train": num_partitions}
        )
        partition = fds.load_partition(partition_id, "train")
        partition = partition.train_test_split(test_size=0.2, seed=42)
        test_set = partition["test"]
        partition = partition["train"]

    partition = partition.with_transform(apply_transforms)
    test_set = test_set.with_transform(apply_transforms)

    trainloader = DataLoader(partition, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(test_set, batch_size=batch_size)

    return trainloader, valloader, config


def get_centralized_testloader(dataset_name: str, batch_size: int) -> DataLoader:
    """Generate the dataloader for the centralized test set."""
    _, testloader, _ = get_dataloaders(
        dataset_name, partition_id=0, num_partitions=1, batch_size=batch_size
    )
    return testloader


# ---------------------------------
# Training/Testing and Weight Handling
# ---------------------------------
def train(net: nn.Module, trainloader: DataLoader, epochs: int, config: Dict):
    """Train the network on the training set."""
    img_key, label_key = config["img_key"], config["label_key"]
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


def test(net: nn.Module, testloader: DataLoader, config: Dict) -> Tuple[float, float]:
    """Evaluate the network on the test set."""
    img_key, label_key = config["img_key"], config["label_key"]
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

    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be empty")

    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def get_weights(net: nn.Module):
    """Extract model parameters as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: nn.Module, parameters):
    """Apply a list of numpy arrays to a model's parameters."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)