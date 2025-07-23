"""Model and data-related utilities."""
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from flwr_datasets import FederatedDataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model(dataset_name: str) -> nn.Module:
    """Return a ResNet-18 model, adapted for the dataset."""
    num_classes = 62 if dataset_name == "femnist" else 10
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)

    # All datasets are resized to 32x32 and converted to 3 channels,
    # so we use the same model adjustments for all.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    return model.to(DEVICE)

def get_dataloader(dataset_name: str, partition_id: int, num_partitions: int, batch_size: int) -> Tuple[DataLoader, DataLoader, Dict]:
    """Return train/test DataLoaders and config for a given client and dataset."""
    dataset_hub_name = {
        "mnist": "ylecun/mnist",
        "cifar10": "uoft-cs/cifar10",
        "femnist": "flwrlabs/femnist",
    }.get(dataset_name.lower())

    if not dataset_hub_name:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    fds = FederatedDataset(dataset=dataset_hub_name, partitioners={"train": num_partitions})
    partition = fds.load_partition(partition_id, "train")

    # Rename column for CIFAR-10 for consistency
    if dataset_name == "cifar10":
        partition = partition.rename_column("img", "image")
    # Rename column for FEMNIST for consistency
    if dataset_name == "femnist":
        partition = partition.rename_column("character", "label")

    # Define transforms
    pytorch_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    def apply_transforms(batch: Dict) -> Dict:
        """Apply transformations to a batch of data."""
        # The key is now always 'image' after the potential rename.
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    partition = partition.with_transform(apply_transforms)
    train_test_split = partition.train_test_split(test_size=0.2, seed=42)
    trainloader = DataLoader(
        train_test_split["train"], batch_size=batch_size, shuffle=True
    )
    valloader = DataLoader(train_test_split["test"], batch_size=batch_size)
    
    # Create a dummy config (not used in the legacy simulation script)
    config = {"img_key": "image", "label_key": "label"}

    return trainloader, valloader, config


def get_centralized_testloader(dataset_name: str, batch_size: int) -> Tuple[DataLoader, Dict]:
    """Generate the dataloader for the centralized test set."""
    # This function uses get_dataloaders to stay consistent
    _, testloader, config = get_dataloader(
        dataset_name, partition_id=0, num_partitions=1, batch_size=batch_size
    )
    return testloader, config


def train(net: nn.Module, trainloader: DataLoader, epochs: int):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def test(net: nn.Module, testloader: DataLoader) -> Tuple[float, float]:
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0, 0.0
    
    loss /= len(testloader)
    accuracy = correct / total
    return loss, accuracy


def get_parameters(net: nn.Module) -> List[torch.Tensor]:
    """Get model parameters as a list of tensors."""
    return [val.cpu() for _, val in net.state_dict().items()]


def set_parameters(net: nn.Module, parameters: List[torch.Tensor]):
    """Set model parameters from a list of tensors."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: v.clone().to(DEVICE) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)