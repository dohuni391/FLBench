import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import argparse
from utils import get_model, get_partition, get_dataset_info
import numpy as np

def train_one_client(model, loader, epochs, device):
    """Simulates the local training loop for one client."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        for batch in loader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def main():
    """Main function to run the local timing experiment."""
    parser = argparse.ArgumentParser(description="FedAvg Local Computation Timing")
    # Keep the user-friendly choices from the original plan
    parser.add_argument("--dataset", type=str, required=True, choices=["MNIST", "CIFAR-10", "FEMNIST"])
    parser.add_argument("--epochs", type=int, default=5, help="Number of local epochs.")
    parser.add_argument("--clients", type=int, default=100, help="Total number of clients to simulate dataset partitioning.")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs to average for timing.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data and Model ---
    # The get_partition function will handle the name conversion internally
    partition = get_partition(args.dataset, client_id=0, total_clients=args.clients)
    loader = DataLoader(partition, batch_size=32, shuffle=True)
    num_classes = get_dataset_info(args.dataset)
    
    print(f"Dataset: {args.dataset}, Samples/client: {len(partition)}, Classes: {num_classes}")

    timings = []
    for i in range(args.runs):
        print(f"--- Starting Run {i+1}/{args.runs} ---")
        model = get_model(args.dataset, num_classes)
        
        # Warm-up run
        if len(loader) > 0:
            train_one_client(model, loader, 1, device)

        # Timed run
        start_time = time.perf_counter()
        train_one_client(model, loader, args.epochs, device)
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        timings.append(elapsed)
        print(f"Run {i+1} elapsed time: {elapsed:.4f} seconds")

    mean_time = np.mean(timings)
    std_dev = np.std(timings)
    
    print("\n--- Results ---")
    print(f"Ran {args.runs} times for {args.dataset} with {len(partition)} samples.")
    print(f"Local computation for {args.epochs} epochs:")
    print(f"Mean time: {mean_time:.4f} seconds")
    print(f"Standard Deviation: {std_dev:.4f} seconds")
    print("-----------------")
    print("\nThis is the value to fill into Table 1 of your testing plan.")


if __name__ == "__main__":
    main()