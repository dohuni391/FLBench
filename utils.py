import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from typing import Tuple, Dict

# Set the primary device for computation. Prefers CUDA GPU if available.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model(dataset_name: str) -> nn.Module:
    """
    Return a ResNet-18 model tailored for the specified dataset.
    This allows for apples-to-apples timing comparisons as per the plan.
    """
    if dataset_name.lower() in ["mnist", "femnist"]:
        num_classes = 10 if dataset_name.lower() == "mnist" else 62
        model = torchvision.models.resnet18(num_classes=num_classes)
        # Tweak for MNIST/FEMNIST: These have 1-channel (grayscale) images.
        # The original ResNet-18 expects 3-channel (RGB) images, so we replace
        # the first convolutional layer to accept 1 input channel instead of 3.
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif dataset_name.lower() == "cifar10":
        model = torchvision.models.resnet18(num_classes=10)
        # Tweak for CIFAR-10: The images (32x32) are smaller than ImageNet's (224x224).
        # The original 7x7 conv with stride 2 would downsample the image too quickly.
        # We replace it with a 3x3 conv with stride 1 and remove the initial max pooling.
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return model.to(DEVICE)

def get_dataloaders(dataset_name: str, client_id: int, num_clients: int, batch_size: int = 32):
    """
    Return train/test DataLoaders and associated metadata for a given client.
    Handles the specific loading, splitting, and partitioning logic for each dataset.
    """
    dataset_map = {
        "mnist": "ylecun/mnist",
        "cifar10": "uoft-cs/cifar10",
        "femnist": "flwrlabs/femnist"
    }
    dataset_id = dataset_map[dataset_name.lower()]

    # --- Dataset-specific keys and transforms ---
    # Datasets from Hugging Face can have different column names for images and labels.
    # We define them here to make the train/test functions generic.
    if dataset_name.lower() == "femnist":
        img_key = "image"
        label_key = "character" # FEMNIST uses 'character' for its labels.
        pytorch_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    elif dataset_name.lower() == "mnist":
        img_key = "image"
        label_key = "label" # Standard 'label' key.
        pytorch_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    elif dataset_name.lower() == "cifar10":
        img_key = "img" # CIFAR-10 from this source uses 'img'.
        label_key = "label"
        pytorch_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        raise ValueError(f"Unsupported dataset for transforms: {dataset_name}")

    def apply_transforms(batch):
        """Apply the defined PyTorch transformations to a batch of data."""
        batch[img_key] = [pytorch_transforms(img) for img in batch[img_key]]
        return batch

    # --- Dataset loading, splitting, and partitioning ---
    if dataset_name.lower() == "femnist":
        # SPECIAL HANDLING FOR FEMNIST:
        # 1. It only has a 'train' split, so we must manually create a test set.
        # 2. The plan calls for an 80/20 split.
        fds = FederatedDataset(dataset=dataset_id, partitioners={"train": 1}) # Load as one big chunk first.
        full_train_set = fds.load_split("train")
        train_test_split = full_train_set.train_test_split(test_size=0.2, seed=42)
        train_set = train_test_split["train"]
        testset = train_test_split["test"]
        
        # 3. Manually create an IID partition for the current client from the new training set.
        # This simulates the partitioning for timing purposes.
        shard_len = len(train_set) // num_clients
        start_idx = client_id * shard_len
        end_idx = start_idx + shard_len
        partition = train_set.select(range(start_idx, end_idx))
    else:
        # Standard path for MNIST and CIFAR-10 which have pre-defined splits.
        # The FederatedDataset automatically handles IID partitioning.
        fds = FederatedDataset(dataset=dataset_id, partitioners={"train": num_clients})
        partition = fds.load_partition(client_id, "train")
        testset = fds.load_split("test")

    # Apply the transforms to the client's partition and the test set.
    partition.set_transform(apply_transforms)
    testset.set_transform(apply_transforms)
    
    trainloader = DataLoader(partition, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size)
    
    # Return all necessary components.
    return trainloader, testloader, len(partition), img_key, label_key

def train(net, trainloader, epochs, img_key: str, label_key: str, verbose=False):
    """Train the network on the training set using dataset-agnostic keys."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            # Use the dynamic keys to access the correct data from the batch.
            images, labels = batch[img_key].to(DEVICE), batch[label_key].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def test(net, testloader, img_key: str, label_key: str) -> Tuple[float, float]:
    """Evaluate the network on the test set using dataset-agnostic keys."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            # Use the dynamic keys to access the correct data from the batch.
            images, labels = batch[img_key].to(DEVICE), batch[label_key].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy