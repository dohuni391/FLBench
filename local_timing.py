"""Script to time local computation for a single client."""

import argparse
import time
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from utils import get_dataloader, get_model, train, DEVICE


def get_client_dataloader(dataset_name: str, num_clients: int) -> Tuple[DataLoader, int]:
    """
    Returns the dataloader for a single client and number of samples.
    We use client_id=0 as a representative client.
    """
    # Batch size is fixed at 32 as per the plan
    train_loader, _, _ = get_dataloader(
        dataset_name=dataset_name,
        partition_id=0,
        num_partitions=num_clients,
        batch_size=32,
    )
    num_samples = len(train_loader.dataset)
    return train_loader, num_samples


def main():
    parser = argparse.ArgumentParser(description="Local Computation Timing")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mnist", "cifar10", "femnist"],
        help="Dataset to use for timing.",
    )
    parser.add_argument(
        "--clients",
        type=int,
        required=True,
        help="Total number of clients to simulate data partitioning.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of local epochs to run.",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of runs to average timing over."
    )
    args = parser.parse_args()

    print(f"Timing local computation for:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Client partition simulation: 1 of {args.clients} clients")
    print(f"  Local epochs: {args.epochs}")
    print(f"  Averaging over {args.runs} runs")

    # Get data for one client
    train_loader, num_samples = get_client_dataloader(args.dataset, args.clients)
    print(f"  Samples per client: {num_samples}")

    # Initialize model
    model = get_model(args.dataset)
    device = DEVICE
    print(f"  Using device: {device}")

    total_time = 0.0
    for i in range(args.runs):
        print(f"\n--- Run {i+1}/{args.runs} ---")
        # Re-initialize model to start from scratch for a fair timing run
        model = get_model(args.dataset)
        
        start_time = time.perf_counter()
        train(model, train_loader, epochs=args.epochs)
        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        print(f"Run {i+1} elapsed time: {elapsed_time:.4f} seconds")

    avg_time = total_time / args.runs
    print("\n--- Results ---")
    print(f"Average time over {args.runs} runs: {avg_time:.4f} seconds")


if __name__ == "__main__":
    main()