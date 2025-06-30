import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from flwr_datasets import FederatedDataset
import argparse
import time

# Ensure reproducibility
torch.manual_seed(42)

def get_model(dataset_name: str, num_classes: int) -> nn.Module:
    """
    Returns a ResNet-18 model with modifications for different datasets.
    """
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

def get_huggingface_id(dataset_name: str) -> str:
    """Converts a user-friendly dataset name to its Hugging Face Hub identifier."""
    if dataset_name == "CIFAR-10":
        return "cifar10"
    elif dataset_name == "MNIST":
        return "mnist"
    elif dataset_name == "FEMNIST":
        return "flwrlabs/femnist"
    else:
        # This case should not be hit if using argparse choices, but it's good practice
        raise ValueError(f"Unknown dataset name: {dataset_name}")

def get_partition(dataset_name: str, client_id: int, total_clients: int) -> Dataset:
    """
    Loads and returns a client's partition of the specified dataset.
    """
    dataset_id = get_huggingface_id(dataset_name)
    print(f"Attempting to load partition for identifier: '{dataset_id}' (from user input: '{dataset_name}')")
    
    fds = FederatedDataset(dataset=dataset_id, partitioners={"train": total_clients})
    partition = fds.load_partition(client_id, "train")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def apply_transform(batch):
        image_key = "image" if "image" in batch else "img"
        batch[image_key] = [transform(img) for img in batch[image_key]]
        return batch

    partition = partition.with_transform(apply_transform)
    image_col = "image" if "image" in partition.features else "img"
    partition.set_format(type="torch", columns=[image_col, "label"])
    if image_col == "img":
        partition = partition.rename_column("img", "image")
    return partition

def get_full_testset(dataset_name: str):
    """
    Loads and returns the full test set for a given dataset.
    """
    dataset_id = get_huggingface_id(dataset_name)
    print(f"Attempting to load full test set for identifier: '{dataset_id}' (from user input: '{dataset_name}')")
    
    fds = FederatedDataset(dataset=dataset_id, partitioners={"train": 1}) # Dummy partitioner
    test_set = fds.load_split("test")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def apply_transform(batch):
        image_key = "image" if "image" in batch else "img"
        batch[image_key] = [transform(img) for img in batch[image_key]]
        return batch

    test_set = test_set.with_transform(apply_transform)
    image_col = "image" if "image" in test_set.features else "img"
    test_set.set_format(type="torch", columns=[image_col, "label"])
    if image_col == "img":
        test_set = test_set.rename_column("img", "image")
    return test_set

def get_dataset_info(dataset_name: str):
    """
    Returns the number of classes for a given dataset.
    """
    if dataset_name in ["MNIST", "CIFAR-10"]:
        return 10
    elif dataset_name == "FEMNIST":
        return 62
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")