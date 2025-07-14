"""Dataset loading and preparation."""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset

def get_dataloaders(dataset_name: str, client_id: int, num_clients: int, batch_size: int = 32):
    """Return train/test DataLoaders and associated metadata for a given client."""
    dataset_map = {
        "mnist": "ylecun/mnist",
        "cifar10": "uoft-cs/cifar10",
        "femnist": "flwrlabs/femnist"
    }
    dataset_id = dataset_map[dataset_name.lower()]

    # --- Dataset-specific keys and transforms ---
    if dataset_name.lower() in ["mnist", "femnist"]:
        img_key, label_key = "image", "character" if dataset_name.lower() == "femnist" else "label"
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    elif dataset_name.lower() == "cifar10":
        img_key, label_key = "img", "label"
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        raise ValueError(f"Unsupported dataset for transforms: {dataset_name}")

    def apply_transforms(batch):
        # Ensure images are RGB for ResNet-18, even grayscale ones
        batch[img_key] = [transform(img.convert("RGB")) for img in batch[img_key]]
        return batch

    # --- Load, partition, and transform data ---
    if dataset_name.lower() == "femnist":
        # SPECIAL HANDLING FOR FEMNIST
        # 1. It only has a 'train' split. We must manually create a test set.
        fds = FederatedDataset(dataset=dataset_id, partitioners={"train": 1})
        full_dataset = fds.load_split("train")
        
        # 2. Split the dataset into 80% train, 20% test
        train_test_split = full_dataset.train_test_split(test_size=0.2, seed=42)
        train_set = train_test_split["train"]
        testset = train_test_split["test"]
        
        # 3. Partition the new training set among the clients
        partition_size = len(train_set) // num_clients
        start = client_id * partition_size
        end = start + partition_size
        partition = train_set.select(range(start, end))
    else:
        fds = FederatedDataset(dataset=dataset_id, partitioners={"train": num_clients})
        partition = fds.load_partition(client_id, "train")
        testset = fds.load_split("test")

    partition.set_transform(apply_transforms)
    testset.set_transform(apply_transforms)

    trainloader = DataLoader(partition, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size)

    return trainloader, testloader, img_key, label_key

def get_centralized_testloader(dataset_name: str):
    """Generate the dataloader for the centralized test set."""
    _, testloader, _, _ = get_dataloaders(dataset_name, 0, 1)
    return testloader